"""
PPO v4 — Dead-Reckoning Augmented + Dense Reward Shaping
==========================================================
The network LEARNS everything. No hardcoded behaviors.

Key idea: feed dead-reckoned (x, y, heading) into the observation.
The network can then learn:
  - "I'm at x=50 → boundary is LEFT → after attachment, turn left and push"
  - "I've been going FW for 70 steps → time to turn (systematic search)"
  - "Stuck + sensors = wall → turn away"

Observation (34 dims):
  [0:18]  raw sensor bits
  [18:23] one-hot prev action
  [23:25] heading as (cos θ, sin θ)
  [24:26] estimated (x, y) normalized to [0,1]  (dead reckoning)
  [26]    distance to nearest boundary (normalized)
  [27]    normalized step count
  [28]    in_push flag
  [29]    stuck flag
  [30]    consecutive stuck / 10 (capped at 1)
  [31]    steps since sensor / 100 (capped at 1)
  [32]    consecutive sensor no push / 30 (capped at 1)
  [33]    wall_suspect flag

Reward shaping (training only):
  - Stuck: -200 → -25 (softened for gradient flow)
  - FW not stuck, no sensor: +0.5 (explore!)
  - FW not stuck, sensors active: +2 (approach box)
  - FW not stuck, push mode: +3 (push toward boundary)
  - Sensor discovery: +5 (found the box!)
  - IR contact: +8 (very close to box)
  - FW into stuck (new): -15 (learn to avoid walls)
  - Turn clearing stuck: +8 (learn unwedge pattern)
  - Push + getting closer to boundary: +4 (learn push direction)
  - Push + getting farther from boundary: -2 (wrong way!)
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
ACTION_ANGLES = [45.0, 22.5, 0.0, -22.5, -45.0]
STEP_SIZE = 5.0
ARENA_MIN = 28.0
ARENA_MAX = 472.0
NUM_ACTIONS = 5
OBS_DIM = 35


# ═══════════════════════════════════════════
# Dead Reckoning
# ═══════════════════════════════════════════
class DR:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.x = 250.0
        self.y = 250.0
        self.heading = 0.0
        self.prev_dist_to_boundary = self._dist()
    
    def _dist(self):
        return min(self.x - ARENA_MIN, ARENA_MAX - self.x,
                   self.y - ARENA_MIN, ARENA_MAX - self.y)
    
    def update(self, action_idx, stuck):
        angle = ACTION_ANGLES[action_idx]
        if angle != 0:
            self.heading = (self.heading + angle) % 360
        elif not stuck:
            rad = math.radians(self.heading)
            self.x += STEP_SIZE * math.cos(rad)
            self.y += STEP_SIZE * math.sin(rad)
            self.x = max(ARENA_MIN, min(ARENA_MAX, self.x))
            self.y = max(ARENA_MIN, min(ARENA_MAX, self.y))
        else:
            # Stuck = at boundary, calibrate
            rad = math.radians(self.heading)
            dx, dy = math.cos(rad), math.sin(rad)
            if abs(dx) > abs(dy):
                self.x = ARENA_MAX if dx > 0 else ARENA_MIN
            else:
                self.y = ARENA_MAX if dy > 0 else ARENA_MIN
    
    def boundary_approach_reward(self):
        """Positive if we got closer to boundary, negative if farther."""
        new_dist = self._dist()
        delta = self.prev_dist_to_boundary - new_dist  # positive = closer
        self.prev_dist_to_boundary = new_dist
        return delta / 10.0  # normalize
    
    def features(self):
        """Return dead-reckoning features for observation."""
        rad = math.radians(self.heading)
        return np.array([
            math.cos(rad), math.sin(rad),
            self.x / 500.0, self.y / 500.0,
            self._dist() / 250.0,
        ], dtype=np.float32)


# ═══════════════════════════════════════════
# Mode + Wall Tracking
# ═══════════════════════════════════════════
class Tracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.in_push = False
        self.ir_was_on = False
        self.push_c = 0
        self.consec_stuck = 0
        self.steps_no_sensor = 0
        self.consec_sensor_no_push = 0
        self.wall_suspect = False
        self.wall_cooldown = 0
        self.was_stuck = False
        self.prev_any_sensor = False
    
    def update(self, obs):
        ir = bool(obs[16])
        any_near = any(bool(obs[2*i+1]) for i in range(8))
        any_s = any(bool(obs[j]) for j in range(17))
        stuck = bool(obs[17])
        
        # Push detection
        if not self.in_push:
            if ir and any_near: self.in_push = True
            if self.ir_was_on and any_s:
                self.push_c += 1
                if self.push_c >= 2: self.in_push = True
            elif any_s: self.push_c += 1
            else: self.push_c = 0
        self.ir_was_on = ir
        
        # Stuck
        self.was_stuck = stuck
        if stuck:
            self.consec_stuck += 1
            if any_s:
                self.wall_suspect = True
                self.wall_cooldown = 15
        else:
            self.consec_stuck = 0
        
        # Sensor tracking
        if any_s and not self.in_push:
            self.consec_sensor_no_push += 1
        else:
            self.consec_sensor_no_push = 0
        
        if not any_s and not stuck:
            self.wall_cooldown = max(0, self.wall_cooldown - 1)
            if self.wall_cooldown == 0:
                self.wall_suspect = False
        
        self.steps_no_sensor = 0 if any_s else self.steps_no_sensor + 1
        self.prev_any_sensor = any_s
    
    def features(self):
        return np.array([
            float(self.in_push),
            float(self.was_stuck),
            min(self.consec_stuck / 10.0, 1.0),
            min(self.steps_no_sensor / 100.0, 1.0),
            min(self.consec_sensor_no_push / 30.0, 1.0),
            float(self.wall_suspect),
        ], dtype=np.float32)


# ═══════════════════════════════════════════
# Observation builder
# ═══════════════════════════════════════════
def build_obs(raw, prev_action, step, max_steps, dr, tracker):
    """34-dim observation."""
    o = np.array(raw, dtype=np.float32)
    ah = np.zeros(5, dtype=np.float32)
    if prev_action >= 0: ah[prev_action] = 1.0
    step_norm = np.array([step / max_steps], dtype=np.float32)
    return np.concatenate([o, ah, dr.features(), step_norm, tracker.features()])


# ═══════════════════════════════════════════
# Reward shaping
# ═══════════════════════════════════════════
def shape_reward(raw_r, obs, action, tracker, dr, prev_stuck, prev_any_sensor):
    stuck = bool(obs[17])
    any_s = any(obs[j] for j in range(17))
    ir = bool(obs[16])
    
    r = raw_r
    
    # Soften stuck: -200 → -25
    if stuck:
        r += 175.0
    
    # FW exploration bonus
    if action == 2 and not stuck:
        if not any_s:
            r += 0.5  # explore
        elif not tracker.in_push:
            r += 2.0  # approach box
        else:
            r += 3.0  # push
    
    # Sensor discovery
    if any_s and not prev_any_sensor:
        r += 5.0
    
    # IR contact
    if ir and not tracker.in_push:
        r += 8.0
    
    # FW into new stuck
    if stuck and action == 2 and not prev_stuck:
        r -= 15.0
    
    # Successfully escaped stuck
    if prev_stuck and not stuck:
        r += 8.0
    
    # Push direction reward: closer to boundary = good
    if tracker.in_push and action == 2 and not stuck:
        approach = dr.boundary_approach_reward()
        if approach > 0:
            r += 4.0 * approach  # getting closer
        else:
            r -= 2.0 * abs(approach)  # getting farther
    
    return r / 50.0


# ═══════════════════════════════════════════
# GRU Actor-Critic
# ═══════════════════════════════════════════
class GRUActorCritic(nn.Module):
    def __init__(self, input_dim=OBS_DIM, gru_hidden=128, n_actions=5):
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
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
    
    def forward(self, obs, hx=None):
        if obs.dim() == 2: obs = obs.unsqueeze(1)
        B = obs.shape[0]
        if hx is None: hx = torch.zeros(1, B, self.gru_hidden, device=obs.device)
        x = self.fc_in(obs)
        out, hx_new = self.gru(x, hx)
        feat = out[:, -1, :]
        return self.actor(feat), self.critic(feat).squeeze(-1), hx_new
    
    def get_action(self, obs_np, hx, deterministic=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
            logits, value, hx_new = self.forward(obs_t, hx)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax(-1).item()
            else:
                action = torch.distributions.Categorical(probs).sample().item()
            lp = torch.log(probs[0, action] + 1e-8).item()
        return action, lp, value.item(), hx_new


# ═══════════════════════════════════════════
# Episode buffer
# ═══════════════════════════════════════════
class EpBuffer:
    def __init__(self):
        self.eps = []
        self._c = {"o":[],"a":[],"lp":[],"r":[],"v":[],"d":[]}
    
    def add(self, o, a, lp, r, v, d):
        self._c["o"].append(o); self._c["a"].append(a); self._c["lp"].append(lp)
        self._c["r"].append(r); self._c["v"].append(v); self._c["d"].append(d)
    
    def end_ep(self, last_v):
        self._c["lv"] = last_v; self.eps.append(self._c)
        self._c = {"o":[],"a":[],"lp":[],"r":[],"v":[],"d":[]}
    
    def compute(self, gamma=0.99, lam=0.95):
        ao, aa, alp, ar, aad = [],[],[],[],[]
        for ep in self.eps:
            n = len(ep["r"])
            if n == 0: continue
            rew, val, dones = np.array(ep["r"]), np.array(ep["v"]), np.array(ep["d"])
            adv = np.zeros(n, np.float32); gae = 0
            for t in reversed(range(n)):
                nv = ep["lv"] if t == n-1 else val[t+1]
                delta = rew[t] + gamma*(1-dones[t])*nv - val[t]
                gae = delta + gamma*lam*(1-dones[t])*gae
                adv[t] = gae
            ao.extend(ep["o"]); aa.extend(ep["a"]); alp.extend(ep["lp"])
            ar.extend((adv + val).tolist()); aad.extend(adv.tolist())
        o = torch.FloatTensor(np.array(ao))
        a = torch.LongTensor(aa); lp = torch.FloatTensor(alp)
        r = torch.FloatTensor(ar); ad = torch.FloatTensor(aad)
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        return o, a, lp, r, ad
    
    def steps(self):
        return sum(len(e["r"]) for e in self.eps) + len(self._c["r"])
    
    def clear(self):
        self.eps = []; self._c = {"o":[],"a":[],"lp":[],"r":[],"v":[],"d":[]}


def ppo_update(model, opt, o, a, olp, ret, adv, clip=0.2, ent_c=0.02, val_c=0.5, epochs=4, bs=256):
    n = len(o)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            e = min(s+bs, n)
            i = idx[s:e]
            logits, vals, _ = model(o[i])
            p = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(p)
            nlp = dist.log_prob(a[i]); ent = dist.entropy().mean()
            ratio = torch.exp(nlp - olp[i])
            s1 = ratio * adv[i]; s2 = torch.clamp(ratio, 1-clip, 1+clip) * adv[i]
            loss = -torch.min(s1,s2).mean() + val_c*F.mse_loss(vals,ret[i]) - ent_c*ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()


# ═══════════════════════════════════════════
# Evaluation (uses sampling, not argmax)
# ═══════════════════════════════════════════
def evaluate(model, diff=0, wall=False, n_eps=5, max_steps=2000):
    rews, succs = [], []
    for ep in range(n_eps):
        env = OBELIX(scaling_factor=3, arena_size=500, max_steps=max_steps,
                     wall_obstacles=wall, difficulty=diff, box_speed=2, seed=ep*1000+42)
        raw = env.sensor_feedback.copy()
        dr = DR(); tr = Tracker(); hx = None; prev_a = -1; total_r = 0
        for s in range(max_steps):
            obs = build_obs(raw, prev_a, s, max_steps, dr, tr)
            a, _, _, hx = model.get_action(obs, hx, deterministic=False)
            raw, r, done = env.step(ACTION_LIST[a], render=False)
            dr.update(a, bool(raw[17])); tr.update(raw)
            prev_a = a; total_r += r
            if done: break
        rews.append(total_r); succs.append(int(env.enable_push and env.done))
    return np.mean(rews), np.std(rews), np.mean(succs)


# ═══════════════════════════════════════════
# Training
# ═══════════════════════════════════════════
def train(args):
    print("=" * 60)
    print("PPO v4 — Dead-Reckoning Augmented (all learned)")
    print(f"OBS_DIM={OBS_DIM}, GRU_HIDDEN={args.gru_hidden}")
    print("=" * 60)
    
    model = GRUActorCritic(input_dim=OBS_DIM, gru_hidden=args.gru_hidden)
    opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    best_eval = -float("inf")
    all_rews = []; total_steps = 0; iteration = 0
    
    # Schedule: 65% wall episodes
    schedule = []
    if args.curriculum:
        for diff in [0, 2, 3]:
            for _ in range(args.episodes_per_level):
                wall = random.random() < 0.65
                schedule.append((diff, wall))
    else:
        schedule = [(args.difficulty, args.wall_obstacles)] * args.n_episodes
    random.shuffle(schedule)
    
    n_total = len(schedule)
    ep_idx = 0
    
    # Teacher: simple reactive, anneals to 0
    def teacher_action(obs, consec_stuck, stuck_dir, steps_no_s, in_push):
        o = np.asarray(obs, dtype=int)
        stuck = bool(o[17]); ir = bool(o[16])
        left = bool(o[0] or o[1] or o[2] or o[3])
        front = bool(o[4] or o[5] or o[6] or o[7] or o[8] or o[9] or o[10] or o[11])
        right = bool(o[12] or o[13] or o[14] or o[15])
        if stuck:
            c = consec_stuck % 3
            if c < 2: return 0 if stuck_dir == 0 else 4
            return 2
        if in_push:
            if ir or front: return 2
            if left: return 1
            if right: return 3
            return 2
        if ir: return 2
        if front: return 2
        if left: return 1
        if right: return 3
        r = random.random()
        if r < 0.65: return 2
        elif r < 0.80: return 1
        elif r < 0.95: return 3
        else: return random.choice([0, 4])
    
    while ep_idx < n_total:
        buf = EpBuffer()
        ep_rews = []
        
        while buf.steps() < args.steps_per_iter and ep_idx < n_total:
            diff, wall = schedule[ep_idx]; ep_idx += 1
            seed = random.randint(0, 1_000_000)
            env = OBELIX(scaling_factor=3, arena_size=500, max_steps=args.max_steps,
                         wall_obstacles=wall, difficulty=diff, box_speed=2, seed=seed)
            raw = env.sensor_feedback.copy()
            
            dr = DR(); tr = Tracker(); hx = None; prev_a = -1
            ep_rew = 0; prev_stuck = False; prev_any_s = False
            consec_stuck = 0; stuck_dir = 0; steps_no_s = 0
            
            frac = min(1.0, ep_idx / (n_total * args.teacher_anneal))
            p_teacher = args.teacher_prob * (1 - frac)
            
            for s in range(args.max_steps):
                obs = build_obs(raw, prev_a, s, args.max_steps, dr, tr)
                
                if random.random() < p_teacher:
                    a = teacher_action(raw, consec_stuck, stuck_dir, steps_no_s, tr.in_push)
                    _, lp, val, hx_new = model.get_action(obs, hx)
                else:
                    a, lp, val, hx_new = model.get_action(obs, hx)
                
                raw_new, r, done = env.step(ACTION_LIST[a], render=False)
                shaped_r = shape_reward(r, raw_new, a, tr, dr, prev_stuck, prev_any_s)
                
                stuck = bool(raw_new[17])
                any_s = any(raw_new[j] for j in range(17))
                dr.update(a, stuck); tr.update(raw_new)
                buf.add(obs, a, lp, shaped_r, val, float(done))
                
                if stuck:
                    consec_stuck += 1
                    if consec_stuck % 9 == 0: stuck_dir = 1 - stuck_dir
                else:
                    consec_stuck = 0
                steps_no_s = 0 if any_s else steps_no_s + 1
                
                prev_stuck = stuck; prev_any_s = any_s
                prev_a = a; raw = raw_new.copy()
                ep_rew += r; total_steps += 1; hx = hx_new
                
                if done: break
            
            _, _, lv, _ = model.get_action(obs, hx)
            buf.end_ep(lv)
            all_rews.append(ep_rew); ep_rews.append(ep_rew)
        
        # PPO update
        o, a, lp, ret, adv = buf.compute(gamma=args.gamma, lam=args.gae_lambda)
        frac2 = min(1.0, total_steps / (n_total * args.max_steps * 0.7))
        ent_c = args.entropy_coef * (1 - frac2) + 0.015 * frac2
        ppo_update(model, opt, o, a, lp, ret, adv, clip=args.clip,
                   ent_c=ent_c, val_c=args.value_coef, epochs=args.n_epochs, bs=args.batch_size)
        buf.clear(); iteration += 1
        
        # Log
        if iteration % 2 == 0:
            mr = np.mean(ep_rews) if ep_rews else 0
            print(f"Ep {ep_idx:5d}/{n_total} | Steps {total_steps:8d} | "
                  f"R={mr:8.1f} | ent={ent_c:.4f} | teacher={p_teacher:.3f}")
        
        # Eval
        if iteration % args.eval_every == 0:
            for d in [0, 3]:
                for w in [False, True]:
                    er, es, esr = evaluate(model, d, w, n_eps=3)
                    print(f"  EVAL d={d} w={w}: R={er:.0f}±{es:.0f} SR={esr:.0%}")
            er0, _, _ = evaluate(model, 0, False, n_eps=5)
            if er0 > best_eval:
                best_eval = er0
                torch.save(model.state_dict(), args.save_path)
                print(f"  >> Saved best (R={er0:.0f})")
    
    torch.save(model.state_dict(), args.save_path.replace(".pth", "_final.pth"))
    np.savez(args.save_path.replace(".pth", "_log.npz"), ep_rewards=np.array(all_rews))
    
    print("\n" + "=" * 60)
    print("Final Evaluation (5 eps each)")
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
    p.add_argument("--teacher_prob", type=float, default=0.25)
    p.add_argument("--teacher_anneal", type=float, default=0.35)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--save_path", type=str, default="weights_ppo_v4.pth")
    args = p.parse_args()
    train(args)