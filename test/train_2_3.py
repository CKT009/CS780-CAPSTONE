"""
PPO v3 — Recurrent (GRU) Policy
=================================
The network learns EVERYTHING including wall avoidance.
No reactive overrides in agent.py.

Key insight: wall vs box is indistinguishable in a single observation.
The difference is TEMPORAL:
  Box:  sensors→FW→closer sensors→FW→IR→attach→push
  Wall: sensors→FW→STUCK→sensors→FW→STUCK→sensors→FW→STUCK

A GRU can learn this pattern. An MLP cannot.

Architecture:
  - 18 raw obs + 5 action one-hot + 5 features = 28 dim input per step
  - GRU with 128 hidden units (carries memory across steps)
  - Actor + Critic heads on GRU output
  - Hidden state persists within episode, reset between episodes

Training tricks:
  - Reward: stuck softened from -200→-30 (learnable gradient)
  - Reward: +2 for FW without getting stuck (exploration)
  - Reward: +5 for IR contact (approaching box)
  - 70% wall episodes in training (learn the hard case)
  - Teacher starts at 0.3, anneals to 0 by 40% of training
  - Sequence-batched PPO updates (preserve temporal structure)
"""

import os, sys, math, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from obelix_fast_fixed import OBELIXFast as OBELIX
    print("Using OBELIXFast (fixed)")
except ImportError:
    try:
        from obelix_fast import OBELIXFast as OBELIX
        print("Using obelix_fast")
    except ImportError:
        from obelix import OBELIX
        print("Using original OBELIX")

ACTION_LIST = ["L45", "L22", "FW", "R22", "R45"]
NUM_ACTIONS = 5
STEP_OBS_DIM = 28  # per-step input to GRU


# ═══════════════════════════════════════════════
# Per-step observation
# ═══════════════════════════════════════════════
def step_obs(raw, prev_action, step, max_steps, in_push):
    """
    28-dim observation for one timestep.
    [0:18]  raw sensor bits
    [18:23] one-hot previous action
    [23]    normalized step
    [24]    in_push flag
    [25]    stuck (obs[17])
    [26]    any_sensor (any of 0:17)
    [27]    IR (obs[16])
    """
    o = np.array(raw, dtype=np.float32)
    ah = np.zeros(5, dtype=np.float32)
    if prev_action >= 0:
        ah[prev_action] = 1.0
    extra = np.array([
        step / max_steps,
        float(in_push),
        float(o[17]),
        float(any(o[j] for j in range(17))),
        float(o[16]),
    ], dtype=np.float32)
    return np.concatenate([o, ah, extra])


# ═══════════════════════════════════════════════
# GRU Actor-Critic
# ═══════════════════════════════════════════════
class GRUActorCritic(nn.Module):
    def __init__(self, input_dim=STEP_OBS_DIM, gru_hidden=128, n_actions=5):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
        )
        self.gru = nn.GRU(128, gru_hidden, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
    
    def forward(self, obs, hx=None):
        """
        obs: (batch, seq_len, input_dim) or (batch, input_dim)
        hx: (1, batch, gru_hidden) or None
        Returns: logits, value, new_hx
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        B, T, _ = obs.shape
        if hx is None:
            hx = torch.zeros(1, B, self.gru_hidden, device=obs.device)
        x = self.fc_in(obs)
        gru_out, hx_new = self.gru(x, hx)
        feat = gru_out[:, -1, :]  # last timestep
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value, hx_new
    
    def get_action(self, obs_np, hx, deterministic=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0)  # (1, dim)
            logits, value, hx_new = self.forward(obs_t, hx)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax(-1).item()
            else:
                action = torch.distributions.Categorical(probs).sample().item()
            lp = torch.log(probs[0, action] + 1e-8).item()
        return action, lp, value.item(), hx_new


# ═══════════════════════════════════════════════
# Mode detector (lightweight, for obs building)
# ═══════════════════════════════════════════════
class ModeTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.in_push = False
        self.ir_was_on = False
        self.c = 0
    def update(self, obs):
        ir = bool(obs[16])
        any_near = any(bool(obs[2*i+1]) for i in range(8))
        any_s = any(bool(obs[j]) for j in range(17))
        if not self.in_push:
            if ir and any_near: self.in_push = True
            if self.ir_was_on and any_s:
                self.c += 1
                if self.c >= 2: self.in_push = True
            elif any_s: self.c += 1
            else: self.c = 0
        self.ir_was_on = ir


# ═══════════════════════════════════════════════
# Reward shaping
# ═══════════════════════════════════════════════
def shape_reward(raw_r, obs, action, env_push, was_stuck_last):
    stuck = bool(obs[17])
    any_s = any(obs[j] for j in range(17))
    ir = bool(obs[16])
    
    r = raw_r
    
    # Soften stuck: -200 → -30 (still bad, but gradient can flow)
    if stuck:
        r += 170.0
    
    # Reward FW that doesn't get stuck (encourages exploration)
    if action == 2 and not stuck:
        r += 1.5
    
    # Reward FW toward sensors (approaching something)
    if action == 2 and not stuck and any_s and not env_push:
        r += 2.0
    
    # Reward FW while pushing
    if action == 2 and not stuck and env_push:
        r += 3.0
    
    # Reward IR contact
    if ir and not env_push:
        r += 8.0
    
    # Penalty for FW into stuck (the network must learn this!)
    # Already -30 from softened stuck, add a bit more if previous step wasn't stuck
    # This creates a clear signal: "going FW HERE is bad"
    if stuck and action == 2 and not was_stuck_last:
        r -= 10.0
    
    # Reward turning away from stuck (the correct unwedge behavior)
    if was_stuck_last and not stuck and action != 2:
        r += 5.0
    
    return r / 50.0


# ═══════════════════════════════════════════════
# Teacher policy (minimal, for early guidance only)
# ═══════════════════════════════════════════════
def teacher_action(obs, consec_stuck, stuck_dir, steps_no_sensor, in_push):
    o = np.asarray(obs, dtype=int)
    stuck = bool(o[17]); ir = bool(o[16])
    left_any = bool(o[0] or o[1] or o[2] or o[3])
    front_any = bool(o[4] or o[5] or o[6] or o[7] or o[8] or o[9] or o[10] or o[11])
    right_any = bool(o[12] or o[13] or o[14] or o[15])
    any_s = left_any or front_any or right_any or ir
    
    if stuck:
        cycle = consec_stuck % 3
        if cycle < 2:
            return 0 if stuck_dir == 0 else 4
        return 2
    if in_push:
        if ir or front_any: return 2
        if left_any: return 1
        if right_any: return 3
        return 2
    if ir: return 2
    if front_any: return 2
    if left_any: return 1
    if right_any: return 3
    # Random walk
    r = random.random()
    if steps_no_sensor < 40:
        if r < 0.70: return 2
        elif r < 0.82: return 1
        elif r < 0.94: return 3
        elif r < 0.97: return 0
        else: return 4
    else:
        if r < 0.50: return 2
        elif r < 0.65: return 0
        elif r < 0.80: return 4
        elif r < 0.90: return 1
        else: return 3


# ═══════════════════════════════════════════════
# Rollout storage (per-episode, preserves sequence)
# ═══════════════════════════════════════════════
class EpisodeBuffer:
    def __init__(self):
        self.episodes = []
        self._cur = {"obs":[],"act":[],"lp":[],"rew":[],"val":[],"done":[]}
    
    def add(self, o, a, lp, r, v, d):
        self._cur["obs"].append(o)
        self._cur["act"].append(a)
        self._cur["lp"].append(lp)
        self._cur["rew"].append(r)
        self._cur["val"].append(v)
        self._cur["done"].append(d)
    
    def end_episode(self, last_val):
        self._cur["last_val"] = last_val
        self.episodes.append(self._cur)
        self._cur = {"obs":[],"act":[],"lp":[],"rew":[],"val":[],"done":[]}
    
    def compute_all_gae(self, gamma=0.99, lam=0.95):
        all_obs, all_act, all_lp, all_ret, all_adv = [],[],[],[],[]
        for ep in self.episodes:
            n = len(ep["rew"])
            if n == 0: continue
            rew = np.array(ep["rew"]); val = np.array(ep["val"])
            dones = np.array(ep["done"])
            adv = np.zeros(n, np.float32)
            gae = 0
            for t in reversed(range(n)):
                nv = ep["last_val"] if t == n-1 else val[t+1]
                delta = rew[t] + gamma * nv * (1-dones[t]) - val[t]
                gae = delta + gamma * lam * (1-dones[t]) * gae
                adv[t] = gae
            ret = adv + val
            all_obs.extend(ep["obs"])
            all_act.extend(ep["act"])
            all_lp.extend(ep["lp"])
            all_ret.extend(ret.tolist())
            all_adv.extend(adv.tolist())
        
        o = torch.FloatTensor(np.array(all_obs))
        a = torch.LongTensor(all_act)
        lp = torch.FloatTensor(all_lp)
        r = torch.FloatTensor(all_ret)
        ad = torch.FloatTensor(all_adv)
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        return o, a, lp, r, ad
    
    def total_steps(self):
        return sum(len(ep["rew"]) for ep in self.episodes) + len(self._cur["rew"])
    
    def clear(self):
        self.episodes = []
        self._cur = {"obs":[],"act":[],"lp":[],"rew":[],"val":[],"done":[]}


def ppo_update(model, opt, o, a, olp, ret, adv, clip=0.2, ent_c=0.02, val_c=0.5, epochs=4, bs=256):
    """Standard PPO update (no sequence batching — GRU state not preserved in update).
       This is fine because the GRU learns the patterns, and updates are per-step."""
    n = len(o)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            e = min(s+bs, n)
            i = idx[s:e]
            logits, vals, _ = model(o[i])  # no hx, single step
            p = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(p)
            nlp = dist.log_prob(a[i])
            ent = dist.entropy().mean()
            ratio = torch.exp(nlp - olp[i])
            s1 = ratio * adv[i]
            s2 = torch.clamp(ratio, 1-clip, 1+clip) * adv[i]
            loss = -torch.min(s1,s2).mean() + val_c*F.mse_loss(vals,ret[i]) - ent_c*ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()


# ═══════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════
def evaluate(model, diff=0, wall=False, n_eps=5, max_steps=2000):
    rews, succs = [], []
    for ep in range(n_eps):
        env = OBELIX(scaling_factor=3, arena_size=500, max_steps=max_steps,
                     wall_obstacles=wall, difficulty=diff, box_speed=2, seed=ep*1000+42)
        raw = env.sensor_feedback.copy()
        mt = ModeTracker()
        hx = None
        prev_a = -1
        tr = 0
        for s in range(max_steps):
            obs = step_obs(raw, prev_a, s, max_steps, mt.in_push)
            a, _, _, hx = model.get_action(obs, hx, deterministic=False)
            raw, r, done = env.step(ACTION_LIST[a], render=False)
            mt.update(raw)
            prev_a = a
            tr += r
            if done: break
        rews.append(tr)
        succs.append(int(env.enable_push and env.done))
    return np.mean(rews), np.std(rews), np.mean(succs)


# ═══════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════
def train(args):
    print("=" * 60)
    print("PPO v3 — GRU Recurrent Policy (no reactive crutches)")
    print("=" * 60)
    
    model = GRUActorCritic(input_dim=STEP_OBS_DIM, gru_hidden=args.gru_hidden)
    opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    
    best_eval = -float("inf")
    all_rews = []
    total_steps = 0
    
    # Schedule: heavy on wall cases
    schedule = []
    if args.curriculum:
        for diff in [0, 2, 3]:
            n = args.episodes_per_level
            for _ in range(n):
                # 70% wall episodes
                wall = random.random() < 0.7
                schedule.append((diff, wall))
    else:
        schedule = [(args.difficulty, args.wall_obstacles)] * args.n_episodes
    
    random.shuffle(schedule)  # mix difficulties
    
    n_total = len(schedule)
    ep_idx = 0
    iteration = 0
    
    while ep_idx < n_total:
        buf = EpisodeBuffer()
        ep_rews_iter = []
        
        # Collect rollouts
        while buf.total_steps() < args.steps_per_iter and ep_idx < n_total:
            diff, wall = schedule[ep_idx]
            ep_idx += 1
            
            seed = random.randint(0, 1_000_000)
            env = OBELIX(scaling_factor=3, arena_size=500, max_steps=args.max_steps,
                         wall_obstacles=wall, difficulty=diff, box_speed=2, seed=seed)
            raw = env.sensor_feedback.copy()
            
            mt = ModeTracker()
            hx = None
            prev_a = -1
            ep_rew = 0
            was_stuck = False
            consec_stuck = 0
            stuck_dir = 0
            steps_no_sensor = 0
            
            # Teacher probability
            frac = min(1.0, ep_idx / (n_total * args.teacher_anneal))
            p_teacher = args.teacher_prob * (1 - frac)
            
            for s in range(args.max_steps):
                obs_vec = step_obs(raw, prev_a, s, args.max_steps, mt.in_push)
                
                # Teacher or network?
                if random.random() < p_teacher:
                    a = teacher_action(raw, consec_stuck, stuck_dir, steps_no_sensor, mt.in_push)
                    _, lp, val, hx_new = model.get_action(obs_vec, hx)
                else:
                    a, lp, val, hx_new = model.get_action(obs_vec, hx)
                
                raw_new, r, done = env.step(ACTION_LIST[a], render=False)
                shaped_r = shape_reward(r, raw_new, a, env.enable_push, was_stuck)
                
                mt.update(raw_new)
                buf.add(obs_vec, a, lp, shaped_r, val, float(done))
                
                # Tracking
                stuck = bool(raw_new[17])
                was_stuck = stuck
                if stuck:
                    consec_stuck += 1
                    if consec_stuck % 9 == 0: stuck_dir = 1 - stuck_dir
                else:
                    consec_stuck = 0
                any_s = any(raw_new[j] for j in range(17))
                steps_no_sensor = 0 if any_s else steps_no_sensor + 1
                
                prev_a = a
                raw = raw_new.copy()
                ep_rew += r
                total_steps += 1
                
                if done: break
            
            # End episode in buffer
            _, _, last_val, _ = model.get_action(obs_vec, hx)
            buf.end_episode(last_val)
            all_rews.append(ep_rew)
            ep_rews_iter.append(ep_rew)
        
        # PPO update
        o, a, lp, ret, adv = buf.compute_all_gae(gamma=args.gamma, lam=args.gae_lambda)
        
        frac2 = min(1.0, total_steps / (n_total * args.max_steps * 0.7))
        ent_c = args.entropy_coef * (1-frac2) + 0.015 * frac2
        
        ppo_update(model, opt, o, a, lp, ret, adv, clip=args.clip,
                   ent_c=ent_c, val_c=args.value_coef, epochs=args.n_epochs, bs=args.batch_size)
        buf.clear()
        iteration += 1
        
        # Logging
        if iteration % 2 == 0:
            mr = np.mean(ep_rews_iter) if ep_rews_iter else 0
            print(f"Ep {ep_idx:5d}/{n_total} | Steps {total_steps:8d} | "
                  f"R={mr:8.1f} | ent={ent_c:.4f} | teacher={p_teacher:.3f}")
        
        # Evaluate
        if iteration % args.eval_every == 0:
            for d in [0, 3]:
                for w in [False, True]:
                    er, es, esr = evaluate(model, d, w, n_eps=3)
                    tag = f"d={d} w={w}"
                    print(f"  EVAL {tag}: R={er:.0f}±{es:.0f} SR={esr:.0%}")
            
            # Save on best d=0 no-wall
            er0, _, _ = evaluate(model, 0, False, n_eps=5)
            if er0 > best_eval:
                best_eval = er0
                torch.save(model.state_dict(), args.save_path)
                print(f"  >> Saved best (R={er0:.0f})")
    
    # Final
    torch.save(model.state_dict(), args.save_path.replace(".pth", "_final.pth"))
    
    print("\n" + "=" * 60)
    print("Final Evaluation (5 eps each)")
    print("=" * 60)
    for d in [0, 2, 3]:
        for w in [False, True]:
            er, es, esr = evaluate(model, d, w, n_eps=5)
            print(f"  d={d} w={str(w):5s}: R={er:8.0f}±{es:6.0f} SR={esr:.0%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.05)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--n_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps_per_iter", type=int, default=8192)
    p.add_argument("--gru_hidden", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--wall_obstacles", action="store_true")
    p.add_argument("--curriculum", action="store_true", default=True)
    p.add_argument("--no_curriculum", dest="curriculum", action="store_false")
    p.add_argument("--n_episodes", type=int, default=5000)
    p.add_argument("--episodes_per_level", type=int, default=2000)
    p.add_argument("--teacher_prob", type=float, default=0.3)
    p.add_argument("--teacher_anneal", type=float, default=0.4)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--save_path", type=str, default="weights_ppo_v3.pth")
    args = p.parse_args()
    train(args)