"""
PPO Training for OBELIX — v2
==============================
Hard-won lessons baked in:
  1. stuck_flag only clears on successful FW → turn-turn-FW cycle
  2. No-sensor state → random walk (not learned, just do it)
  3. Reward shaping: soften -200 stuck, bonus for sensor discovery
  4. Teacher-guided exploration: reactive policy acts as "teacher"
     with probability p_teacher (annealed from 0.5 → 0) 
  5. History-augmented obs: last 4 actions + sensor summary
  6. Separate find/push heads (mode-conditioned policy)

Usage:
  python train_ppo.py                          # curriculum, all levels
  python train_ppo.py --difficulty 0           # single level
  python train_ppo.py --episodes_per_level 500 # quick test
"""

import os, sys, math, random, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try fast env first, fall back to original
try:
    from obelix_fast_fixed import OBELIXFast as OBELIX
    print("Using OBELIXFast (fixed)")
except ImportError:
    try:
        from obelix_fast import OBELIXFast as OBELIX
        print("WARNING: using obelix_fast (unfixed far/near bug)")
    except ImportError:
        from obelix import OBELIX
        print("Using original OBELIX (slow)")

ACTION_LIST = ["L45", "L22", "FW", "R22", "R45"]
NUM_ACTIONS = 5


# ═══════════════════════════════════════════════
# Observation Augmentation
# ═══════════════════════════════════════════════
def build_obs(raw_obs, action_hist, step, max_steps, in_push, wall_suspect, consec_stuck):
    """
    Augment 18-bit raw obs → richer feature vector.
    
    Features:
      [0:18]  raw sensor bits
      [18:23] one-hot last action (5)
      [23:28] one-hot second-last action (5)
      [28]    normalized step count
      [29]    in_push flag
      [30]    wall_suspect flag
      [31]    stuck (from obs[17])
      [32]    any_sensor (any of obs[0:17])
      [33]    front_sensor (any of obs[4:12])
      [34]    left_sensor (any of obs[0:4])
      [35]    right_sensor (any of obs[12:16])
      [36]    ir sensor (obs[16])
      [37]    consec_stuck normalized
    Total: 38
    """
    o = np.array(raw_obs, dtype=np.float32)
    
    # Last two actions as one-hot
    a1_oh = np.zeros(5, dtype=np.float32)
    a2_oh = np.zeros(5, dtype=np.float32)
    if len(action_hist) >= 1 and action_hist[-1] >= 0:
        a1_oh[action_hist[-1]] = 1.0
    if len(action_hist) >= 2 and action_hist[-2] >= 0:
        a2_oh[action_hist[-2]] = 1.0
    
    extra = np.array([
        step / max_steps,
        float(in_push),
        float(wall_suspect),
        float(o[17]),                                    # stuck
        float(any(o[j] for j in range(17))),             # any_sensor
        float(any(o[j] for j in range(4, 12))),          # front
        float(any(o[j] for j in range(0, 4))),           # left
        float(any(o[j] for j in range(12, 16))),         # right
        float(o[16]),                                     # IR
        min(consec_stuck / 10.0, 1.0),                   # stuck duration
    ], dtype=np.float32)
    
    return np.concatenate([o, a1_oh, a2_oh, extra])

OBS_DIM = 38


# ═══════════════════════════════════════════════
# Teacher Policy (reactive, hand-coded)
# ═══════════════════════════════════════════════
class TeacherPolicy:
    """
    The reactive policy that we know works.
    Used to guide PPO exploration early in training.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.step = 0
        self.in_push = False
        self.ir_was_on = False
        self.consec = 0
        self.consec_stuck = 0
        self.stuck_dir = 0
        self.wall_suspect = False
        self.prev_action = -1
        self.steps_since_sensor = 0
    
    def update(self, obs):
        """Update internal tracking. Call BEFORE act()."""
        self.step += 1
        ir = bool(obs[16])
        any_near = any(bool(obs[2*i+1]) for i in range(8))
        any_s = any(bool(obs[j]) for j in range(17))
        stuck = bool(obs[17])
        
        # Push detection
        if not self.in_push:
            if ir and any_near:
                self.in_push = True
            if self.ir_was_on and any_s:
                self.consec += 1
                if self.consec >= 2:
                    self.in_push = True
            elif any_s:
                self.consec += 1
            else:
                self.consec = 0
        self.ir_was_on = ir
        
        # Stuck tracking
        if stuck:
            self.consec_stuck += 1
        else:
            self.consec_stuck = 0
        
        # Wall suspect
        if stuck and self.prev_action == 2 and any_s:
            self.wall_suspect = True
        if not any_s and not stuck:
            self.wall_suspect = False
        
        # Steps since sensor
        if any_s:
            self.steps_since_sensor = 0
        else:
            self.steps_since_sensor += 1
    
    def act(self, obs, rng=None):
        """Return action index using reactive rules."""
        o = np.asarray(obs, dtype=int)
        stuck = bool(o[17])
        ir = bool(o[16])
        left_any = bool(o[0] or o[1] or o[2] or o[3])
        front_any = bool(o[4] or o[5] or o[6] or o[7] or o[8] or o[9] or o[10] or o[11])
        right_any = bool(o[12] or o[13] or o[14] or o[15])
        any_sensor = left_any or front_any or right_any or ir
        
        def rand():
            return float(rng.random()) if rng else random.random()
        
        # Stuck: turn-turn-FW cycle
        if stuck:
            cycle = self.consec_stuck % 3
            if self.consec_stuck > 0 and self.consec_stuck % 9 == 0:
                self.stuck_dir = 1 - self.stuck_dir
            if cycle < 2:
                a = 0 if self.stuck_dir == 0 else 4
            else:
                a = 2
            self.prev_action = a
            return a
        
        # Push mode
        if self.in_push:
            if ir or front_any: a = 2
            elif left_any: a = 1
            elif right_any: a = 3
            else: a = 2
            self.prev_action = a
            return a
        
        # Wall suspect
        if self.wall_suspect and any_sensor and not ir:
            if left_any and not right_any: a = 4
            elif right_any and not left_any: a = 0
            else: a = 0 if (self.step % 6 < 3) else 4
            self.prev_action = a
            return a
        
        # Sensor active
        if ir: a = 2
        elif front_any: a = 2
        elif left_any: a = 1
        elif right_any: a = 3
        else:
            # No sensors — random walk
            r = rand()
            if self.steps_since_sensor < 40:
                if r < 0.70: a = 2
                elif r < 0.82: a = 1
                elif r < 0.94: a = 3
                elif r < 0.97: a = 0
                else: a = 4
            elif self.steps_since_sensor < 100:
                if r < 0.55: a = 2
                elif r < 0.70: a = 1
                elif r < 0.85: a = 3
                elif r < 0.92: a = 0
                else: a = 4
            else:
                if r < 0.40: a = 2
                elif r < 0.55: a = 0
                elif r < 0.70: a = 4
                elif r < 0.85: a = 1
                else: a = 3
        
        self.prev_action = a
        return a


# ═══════════════════════════════════════════════
# Mode Detector (for obs augmentation)
# ═══════════════════════════════════════════════
class ModeDetector:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.in_push = False
        self.ir_was_on = False
        self.consec = 0
        self.wall_suspect = False
        self.consec_stuck = 0
        self.prev_action = -1
    
    def update(self, obs, action):
        ir = bool(obs[16])
        any_near = any(bool(obs[2*i+1]) for i in range(8))
        any_s = any(bool(obs[j]) for j in range(17))
        stuck = bool(obs[17])
        
        if not self.in_push:
            if ir and any_near: self.in_push = True
            if self.ir_was_on and any_s:
                self.consec += 1
                if self.consec >= 2: self.in_push = True
            elif any_s: self.consec += 1
            else: self.consec = 0
        self.ir_was_on = ir
        
        if stuck:
            self.consec_stuck += 1
            if self.prev_action == 2 and any_s:
                self.wall_suspect = True
        else:
            self.consec_stuck = 0
        if not any_s and not stuck:
            self.wall_suspect = False
        
        self.prev_action = action


# ═══════════════════════════════════════════════
# Reward Shaping
# ═══════════════════════════════════════════════
def shape_reward(raw_reward, obs, action, env_push, prev_any_sensor, consec_stuck):
    """
    Training-only reward shaping.
    Key: soften -200 stuck penalty (it's too harsh for learning)
    and add small bonuses for productive behavior.
    """
    shaped = raw_reward
    stuck = bool(obs[17])
    any_sensor = any(obs[j] for j in range(17))
    ir = bool(obs[16])
    
    # 1. Soften stuck penalty: -200 → -20 (still negative, but learnable)
    if stuck:
        shaped += 180.0  # raw has -200, net becomes -20
    
    # 2. Bonus for FW when not stuck and no sensors (encourages exploration)
    if action == 2 and not stuck and not any_sensor:
        shaped += 0.3
    
    # 3. Bonus for FW toward sensors (approaching box)
    if action == 2 and not stuck and any_sensor and not env_push:
        shaped += 1.0
    
    # 4. Bonus for FW while pushing
    if action == 2 and not stuck and env_push:
        shaped += 1.5
    
    # 5. Bonus for sensor discovery (first time seeing box)
    if any_sensor and not prev_any_sensor:
        shaped += 3.0
    
    # 6. IR bonus (very close to box)
    if ir and not env_push:
        shaped += 5.0
    
    return shaped / 50.0  # scale down for stable learning


# ═══════════════════════════════════════════════
# Actor-Critic Network
# ═══════════════════════════════════════════════
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=256, n_actions=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
    
    def forward(self, obs):
        feat = self.shared(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value
    
    def get_action(self, obs_np, deterministic=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
            logits, value = self.forward(obs_t)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                action = torch.distributions.Categorical(probs).sample().item()
            log_prob = torch.log(probs[0, action] + 1e-8).item()
        return action, log_prob, value.item()


# ═══════════════════════════════════════════════
# PPO Buffer & Update
# ═══════════════════════════════════════════════
class Buffer:
    def __init__(self):
        self.obs, self.act, self.lp, self.rew, self.val, self.done = [],[],[],[],[],[]
    
    def add(self, o, a, lp, r, v, d):
        self.obs.append(o); self.act.append(a); self.lp.append(lp)
        self.rew.append(r); self.val.append(v); self.done.append(d)
    
    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        n = len(self.rew)
        adv = np.zeros(n, np.float32)
        rew = np.array(self.rew); val = np.array(self.val); dones = np.array(self.done)
        gae = 0
        for t in reversed(range(n)):
            nv = last_val if t == n-1 else val[t+1]
            nd = dones[t]
            delta = rew[t] + gamma * nv * (1 - nd) - val[t]
            gae = delta + gamma * lam * (1 - nd) * gae
            adv[t] = gae
        ret = adv + val
        return ret, adv
    
    def tensors(self, ret, adv):
        o = torch.FloatTensor(np.array(self.obs))
        a = torch.LongTensor(self.act)
        lp = torch.FloatTensor(self.lp)
        r = torch.FloatTensor(ret)
        ad = torch.FloatTensor(adv)
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        return o, a, lp, r, ad
    
    def clear(self):
        self.__init__()


def ppo_update(model, opt, buf, ret, adv, clip=0.2, ent_c=0.02, val_c=0.5, epochs=4, bs=128):
    o, a, olp, r, ad = buf.tensors(ret, adv)
    n = len(o)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            e = min(s+bs, n)
            i = idx[s:e]
            logits, vals = model(o[i])
            p = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(p)
            nlp = dist.log_prob(a[i])
            ent = dist.entropy().mean()
            ratio = torch.exp(nlp - olp[i])
            s1 = ratio * ad[i]
            s2 = torch.clamp(ratio, 1-clip, 1+clip) * ad[i]
            loss = -torch.min(s1, s2).mean() + val_c * F.mse_loss(vals, r[i]) - ent_c * ent
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
                     wall_obstacles=wall, difficulty=diff, box_speed=2,
                     seed=ep*1000+42)
        raw = env.sensor_feedback.copy()
        md = ModeDetector()
        ahist = [-1, -1]
        tr = 0
        for step in range(max_steps):
            aug = build_obs(raw, ahist, step, max_steps, md.in_push, md.wall_suspect, md.consec_stuck)
            
            # For no-sensor state, use random walk instead of network
            any_s = any(raw[j] for j in range(17))
            stuck = bool(raw[17])
            if not any_s and not stuck and not md.in_push:
                rng_val = random.random()
                if rng_val < 0.65: a = 2
                elif rng_val < 0.80: a = 1
                elif rng_val < 0.95: a = 3
                elif rng_val < 0.975: a = 0
                else: a = 4
            else:
                a, _, _ = model.get_action(aug, deterministic=True)
            
            raw, r, done = env.step(ACTION_LIST[a], render=False)
            md.update(raw, a)
            ahist.append(a)
            if len(ahist) > 5: ahist = ahist[-5:]
            tr += r
            if done: break
        rews.append(tr)
        succs.append(int(env.enable_push and env.done))
    return np.mean(rews), np.std(rews), np.mean(succs)


# ═══════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════
def train(args):
    print("=" * 60)
    print("PPO Training v2 — Teacher-Guided + History-Augmented")
    print("=" * 60)
    
    model = ActorCritic(obs_dim=OBS_DIM, hidden=args.hidden)
    opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    teacher = TeacherPolicy()
    
    best_eval = -float("inf")
    all_rews, all_eval = [], []
    total_steps = 0
    
    # Build training schedule
    schedule = []
    if args.curriculum:
        for diff in [0, 2, 3]:
            n = args.episodes_per_level
            for _ in range(n // 2):
                schedule.append((diff, False))
                schedule.append((diff, True))
    else:
        schedule = [(args.difficulty, args.wall_obstacles)] * args.n_episodes
    
    # Teacher probability: starts high, anneals to 0
    p_teacher_start = args.teacher_prob
    p_teacher_end = 0.0
    teacher_anneal_frac = 0.6  # anneal over 60% of training
    
    # Entropy coefficient scheduling
    ent_start = args.entropy_coef
    ent_end = 0.005
    
    ep_idx = 0
    n_total = len(schedule)
    
    while ep_idx < n_total:
        buf = Buffer()
        steps_this_iter = 0
        ep_rews_iter = []
        
        while steps_this_iter < args.steps_per_iter and ep_idx < n_total:
            diff, wall = schedule[ep_idx]
            ep_idx += 1
            
            seed = random.randint(0, 1_000_000)
            env = OBELIX(scaling_factor=3, arena_size=500, max_steps=args.max_steps,
                         wall_obstacles=wall, difficulty=diff, box_speed=2, seed=seed)
            raw = env.sensor_feedback.copy()
            
            teacher.reset()
            md = ModeDetector()
            ahist = [-1, -1]
            ep_rew = 0
            prev_any_sensor = False
            
            # Teacher probability for this episode
            frac = min(1.0, ep_idx / (n_total * teacher_anneal_frac))
            p_teacher = p_teacher_start * (1 - frac) + p_teacher_end * frac
            
            for step in range(args.max_steps):
                aug = build_obs(raw, ahist, step, args.max_steps, 
                               md.in_push, md.wall_suspect, md.consec_stuck)
                
                any_s = any(raw[j] for j in range(17))
                stuck = bool(raw[17])
                
                # Decision: teacher, random walk, or network?
                use_teacher = random.random() < p_teacher
                
                if not any_s and not stuck and not md.in_push:
                    # No sensor state: always random walk (teacher handles this)
                    teacher.update(raw)
                    a = teacher.act(raw)
                    # Still get network's log_prob for PPO (off-policy correction)
                    _, log_prob, value = model.get_action(aug)
                    # Override action but keep value estimate
                    log_prob = 0.0  # won't be used for gradient (importance weight ~0)
                elif use_teacher:
                    # Teacher guides the action
                    teacher.update(raw)
                    a = teacher.act(raw)
                    _, log_prob, value = model.get_action(aug)
                else:
                    # Network acts
                    teacher.update(raw)  # keep teacher state updated
                    a, log_prob, value = model.get_action(aug)
                
                raw_new, r, done = env.step(ACTION_LIST[a], render=False)
                
                # Shape reward for training
                shaped_r = shape_reward(r, raw_new, a, env.enable_push, 
                                       prev_any_sensor, md.consec_stuck)
                
                md.update(raw_new, a)
                ahist.append(a)
                if len(ahist) > 5: ahist = ahist[-5:]
                
                buf.add(aug, a, log_prob, shaped_r, value, float(done))
                
                prev_any_sensor = any_s
                raw = raw_new.copy()
                ep_rew += r  # track unshaped
                steps_this_iter += 1
                total_steps += 1
                
                if done: break
            
            all_rews.append(ep_rew)
            ep_rews_iter.append(ep_rew)
        
        # PPO update
        _, _, last_val = model.get_action(aug)
        ret, adv = buf.compute_gae(last_val, gamma=args.gamma, lam=args.gae_lambda)
        
        frac2 = min(1.0, total_steps / (n_total * args.max_steps * 0.7))
        ent_c = ent_start * (1 - frac2) + ent_end * frac2
        
        ppo_update(model, opt, buf, ret, adv, clip=args.clip,
                   ent_c=ent_c, val_c=args.value_coef,
                   epochs=args.n_epochs, bs=args.batch_size)
        buf.clear()
        
        # Logging
        if ep_idx % args.log_interval < (args.steps_per_iter // args.max_steps + 2):
            mr = np.mean(ep_rews_iter) if ep_rews_iter else 0
            print(f"Ep {ep_idx:5d}/{n_total} | Steps {total_steps:8d} | "
                  f"R={mr:8.1f} | ent={ent_c:.4f} | teacher={p_teacher:.3f}")
        
        # Evaluate
        if ep_idx % args.eval_interval < (args.steps_per_iter // args.max_steps + 2):
            for d in [0, 3]:
                for w in [False, True]:
                    er, es, esr = evaluate(model, d, w, n_eps=3)
                    print(f"  EVAL d={d} w={w}: R={er:.0f}±{es:.0f} SR={esr:.0%}")
                    if d == 0 and not w and er > best_eval:
                        best_eval = er
                        torch.save(model.state_dict(), args.save_path)
                        print(f"  >> Saved best (R={er:.0f})")
    
    # Final save
    torch.save(model.state_dict(), args.save_path.replace(".pth", "_final.pth"))
    np.savez(args.save_path.replace(".pth", "_log.npz"),
             ep_rewards=np.array(all_rews))
    
    # Final eval
    print("\n" + "=" * 60)
    print("Final Evaluation")
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
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps_per_iter", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--wall_obstacles", action="store_true")
    p.add_argument("--curriculum", action="store_true", default=True)
    p.add_argument("--no_curriculum", dest="curriculum", action="store_false")
    p.add_argument("--n_episodes", type=int, default=5000)
    p.add_argument("--episodes_per_level", type=int, default=2000)
    p.add_argument("--teacher_prob", type=float, default=0.5)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--save_path", type=str, default="weights.pth")
    args = p.parse_args()
    train(args)