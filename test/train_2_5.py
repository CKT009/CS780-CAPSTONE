"""
PPO v5 — Root Cause Fix
=========================
WHY previous versions collapsed to "always turn":
  In no-sensor state: FW has 5% chance of -200 (boundary stuck).
  Turns have 0% chance of -200.
  Expected reward: FW ≈ -10, Turn = -1. Network correctly learns Turn > FW.
  → Agent never explores → never finds box → all episodes -2000.

THE FIX (reward shaping):
  No-sensor state: FW without stuck → +3 reward (not -1)
  No-sensor state: FW into stuck → -5 (not -200)
  This makes E[FW] ≈ +2.6, E[Turn] = -1 → network learns FW >> Turn.

  Push state: per-step reward for moving closer to boundary (+5/-3)
  Dead reckoning (x,y,heading) in observation so network can learn direction.

Architecture: GRU with 35-dim obs (same as v4).
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


class DR:
    def __init__(self):
        self.reset()
    def reset(self):
        self.x = 250.0; self.y = 250.0; self.heading = 0.0; self.prev_bdist = self._bdist()
    def _bdist(self):
        return min(self.x - ARENA_MIN, ARENA_MAX - self.x, self.y - ARENA_MIN, ARENA_MAX - self.y)
    def update(self, action_idx, stuck):
        angle = ACTION_ANGLES[action_idx]
        if angle != 0:
            self.heading = (self.heading + angle) % 360
        elif not stuck:
            rad = math.radians(self.heading)
            self.x = max(ARENA_MIN, min(ARENA_MAX, self.x + STEP_SIZE * math.cos(rad)))
            self.y = max(ARENA_MIN, min(ARENA_MAX, self.y + STEP_SIZE * math.sin(rad)))
        else:
            rad = math.radians(self.heading)
            dx, dy = math.cos(rad), math.sin(rad)
            if abs(dx) > abs(dy): self.x = ARENA_MAX if dx > 0 else ARENA_MIN
            else: self.y = ARENA_MAX if dy > 0 else ARENA_MIN
    def boundary_delta(self):
        new = self._bdist()
        delta = self.prev_bdist - new  # positive = closer
        self.prev_bdist = new
        return delta
    def features(self):
        rad = math.radians(self.heading)
        return np.array([math.cos(rad), math.sin(rad), self.x/500, self.y/500, self._bdist()/250], dtype=np.float32)


class Tracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.in_push = False; self.ir_was = False; self.pc = 0
        self.cstuck = 0; self.sns = 0; self.csnp = 0
        self.wsus = False; self.wcd = 0; self.was_stuck = False; self.prev_any = False
    def update(self, obs):
        ir = bool(obs[16]); stuck = bool(obs[17])
        anr = any(bool(obs[2*i+1]) for i in range(8))
        ans = any(bool(obs[j]) for j in range(17))
        if not self.in_push:
            if ir and anr: self.in_push = True
            if self.ir_was and ans:
                self.pc += 1
                if self.pc >= 2: self.in_push = True
            elif ans: self.pc += 1
            else: self.pc = 0
        self.ir_was = ir; self.was_stuck = stuck
        if stuck:
            self.cstuck += 1
            if ans: self.wsus = True; self.wcd = 15
        else: self.cstuck = 0
        if ans and not self.in_push: self.csnp += 1
        else: self.csnp = 0
        if not ans and not stuck:
            self.wcd = max(0, self.wcd - 1)
            if self.wcd == 0: self.wsus = False
        self.sns = 0 if ans else self.sns + 1
        self.prev_any = ans
    def features(self):
        return np.array([float(self.in_push), float(self.was_stuck),
                         min(self.cstuck/10, 1), min(self.sns/100, 1),
                         min(self.csnp/30, 1), float(self.wsus)], dtype=np.float32)


def build_obs(raw, prev_a, step, max_steps, dr, tr):
    o = np.array(raw, dtype=np.float32)
    ah = np.zeros(5, dtype=np.float32)
    if prev_a >= 0: ah[prev_a] = 1.0
    return np.concatenate([o, ah, dr.features(), np.array([step/max_steps], np.float32), tr.features()])


# ═══════════════════════════════════════════
# REWARD SHAPING — the root cause fix
# ═══════════════════════════════════════════
def shape_reward(raw_r, obs, action, tr, dr, prev_stuck, prev_any_s):
    stuck = bool(obs[17])
    any_s = any(obs[j] for j in range(17))
    ir = bool(obs[16])

    # Start with 0, build up from components
    r = 0.0

    # ── PER-STEP BASE ──
    r -= 0.5  # small step cost (encourages efficiency, not as harsh as -1)

    # ── NO-SENSOR STATE: make FW clearly better than turns ──
    if not any_s and not tr.in_push:
        if action == 2:  # FW
            if not stuck:
                r += 3.0    # BIG reward for exploring forward
            else:
                r -= 3.0    # mild penalty for hitting boundary (not -200!)
        else:
            r -= 0.5        # turns are slightly costly when searching
            # BUT: if we were just stuck, turning is GOOD
            if prev_stuck:
                r += 2.0    # reward turning away from boundary

    # ── SENSOR ACTIVE (approaching box or wall) ──
    elif any_s and not tr.in_push:
        if ir:
            r += 10.0       # very close to box! big reward
        elif action == 2 and not stuck:
            r += 3.0        # approaching sensor target
        elif action == 2 and stuck:
            r -= 8.0        # rammed into wall with sensors = bad
        # Sensor discovery bonus
        if any_s and not prev_any_s:
            r += 5.0

    # ── PUSH STATE: reward boundary approach ──
    elif tr.in_push:
        if action == 2 and not stuck:
            delta = dr.boundary_delta()
            if delta > 0:
                r += 8.0    # pushing toward boundary
            else:
                r -= 2.0    # pushing away from boundary
        elif action == 2 and stuck:
            r -= 5.0        # stuck while pushing
        elif not stuck:
            r += 0.5        # turning to reorient (acceptable)

    # ── STUCK ESCAPE PATTERN ──
    if stuck and not any_s:
        # Boundary stuck during search — mild, not catastrophic
        if action != 2:
            r += 1.0        # turning while stuck = good (trying to escape)
        # action == 2 while stuck is handled above

    if prev_stuck and not stuck:
        r += 3.0            # escaped! reward

    # ── ATTACHMENT BONUS ──
    # Raw reward already has +100 for attachment, keep it but scaled
    if raw_r > 50:  # attachment happened
        r += 20.0

    # ── SUCCESS BONUS ──
    if raw_r > 1000:  # success
        r += 100.0

    return r / 20.0  # scale for stable learning


# ═══════════════════════════════════════════
# GRU Actor-Critic (same as v4)
# ═══════════════════════════════════════════
class GRUActorCritic(nn.Module):
    def __init__(self, input_dim=OBS_DIM, gru_hidden=128, n_actions=5):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.fc_in = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.gru = nn.GRU(128, gru_hidden, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Linear(64, n_actions))
        self.critic = nn.Sequential(nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, obs, hx=None):
        if obs.dim() == 2: obs = obs.unsqueeze(1)
        B = obs.shape[0]
        if hx is None: hx = torch.zeros(1, B, self.gru_hidden, device=obs.device)
        out, hx_new = self.gru(self.fc_in(obs), hx)
        feat = out[:, -1, :]
        return self.actor(feat), self.critic(feat).squeeze(-1), hx_new

    def get_action(self, obs_np, hx, det=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
            logits, value, hx_new = self.forward(obs_t, hx)
            probs = F.softmax(logits, dim=-1)
            a = probs.argmax(-1).item() if det else torch.distributions.Categorical(probs).sample().item()
            lp = torch.log(probs[0, a] + 1e-8).item()
        return a, lp, value.item(), hx_new


# ═══════════════════════════════════════════
# Buffer + PPO (same as v4)
# ═══════════════════════════════════════════
class Buf:
    def __init__(self):
        self.eps = []; self._c = {"o":[],"a":[],"lp":[],"r":[],"v":[],"d":[]}
    def add(self, o, a, lp, r, v, d):
        self._c["o"].append(o); self._c["a"].append(a); self._c["lp"].append(lp)
        self._c["r"].append(r); self._c["v"].append(v); self._c["d"].append(d)
    def end(self, lv):
        self._c["lv"] = lv; self.eps.append(self._c)
        self._c = {"o":[],"a":[],"lp":[],"r":[],"v":[],"d":[]}
    def compute(self, gamma=0.99, lam=0.95):
        ao, aa, alp, ar, aad = [],[],[],[],[]
        for ep in self.eps:
            n = len(ep["r"])
            if n == 0: continue
            rew, val, dn = np.array(ep["r"]), np.array(ep["v"]), np.array(ep["d"])
            adv = np.zeros(n, np.float32); gae = 0
            for t in reversed(range(n)):
                nv = ep["lv"] if t == n-1 else val[t+1]
                gae = (rew[t] + gamma*(1-dn[t])*nv - val[t]) + gamma*lam*(1-dn[t])*gae
                adv[t] = gae
            ao.extend(ep["o"]); aa.extend(ep["a"]); alp.extend(ep["lp"])
            ar.extend((adv+val).tolist()); aad.extend(adv.tolist())
        o=torch.FloatTensor(np.array(ao)); a=torch.LongTensor(aa)
        lp=torch.FloatTensor(alp); r=torch.FloatTensor(ar)
        ad=torch.FloatTensor(aad); ad=(ad-ad.mean())/(ad.std()+1e-8)
        return o, a, lp, r, ad
    def steps(self):
        return sum(len(e["r"]) for e in self.eps) + len(self._c["r"])
    def clear(self):
        self.eps=[]; self._c={"o":[],"a":[],"lp":[],"r":[],"v":[],"d":[]}


def ppo_up(model, opt, o, a, olp, ret, adv, clip=0.2, ec=0.02, vc=0.5, ep=4, bs=256):
    n = len(o)
    for _ in range(ep):
        for s in range(0, n, bs):
            e = min(s+bs, n); i = np.random.choice(n, min(bs, n), replace=False)
            lg, v, _ = model(o[i])
            p = F.softmax(lg, -1); d = torch.distributions.Categorical(p)
            nlp = d.log_prob(a[i]); ent = d.entropy().mean()
            ratio = torch.exp(nlp - olp[i])
            loss = -torch.min(ratio*adv[i], torch.clamp(ratio,1-clip,1+clip)*adv[i]).mean() \
                   + vc*F.mse_loss(v, ret[i]) - ec*ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()


def evaluate(model, diff=0, wall=False, n_eps=5, max_steps=2000):
    rews, succs = [], []
    for ep in range(n_eps):
        env = OBELIX(scaling_factor=3, arena_size=500, max_steps=max_steps,
                     wall_obstacles=wall, difficulty=diff, box_speed=2, seed=ep*1000+42)
        raw = env.sensor_feedback.copy()
        dr = DR(); tr = Tracker(); hx = None; pa = -1; total = 0
        for s in range(max_steps):
            obs = build_obs(raw, pa, s, max_steps, dr, tr)
            a, _, _, hx = model.get_action(obs, hx, det=False)
            raw, r, done = env.step(ACTION_LIST[a], render=False)
            dr.update(a, bool(raw[17])); tr.update(raw); pa = a; total += r
            if done: break
        rews.append(total); succs.append(int(env.enable_push and env.done))
    return np.mean(rews), np.std(rews), np.mean(succs)


def train(args):
    print("=" * 60)
    print("PPO v5 — Root Cause Fix (FW reward > Turn reward)")
    print("=" * 60)

    model = GRUActorCritic(input_dim=OBS_DIM, gru_hidden=args.gru_hidden)
    opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    best = -1e9; total_steps = 0; iteration = 0

    schedule = []
    if args.curriculum:
        for diff in [0, 2, 3]:
            for _ in range(args.episodes_per_level):
                schedule.append((diff, random.random() < 0.65))
    else:
        schedule = [(args.difficulty, args.wall_obstacles)] * args.n_episodes
    random.shuffle(schedule)
    n_total = len(schedule); ep_idx = 0

    # Teacher
    def t_act(obs, cs, sd, sn, ip):
        o = np.asarray(obs, dtype=int)
        st = bool(o[17]); ir = bool(o[16])
        l = bool(o[0]or o[1]or o[2]or o[3]); f = bool(o[4]or o[5]or o[6]or o[7]or o[8]or o[9]or o[10]or o[11])
        ri = bool(o[12]or o[13]or o[14]or o[15])
        if st:
            c = cs%3
            if c<2: return 0 if sd==0 else 4
            return 2
        if ip:
            if ir or f: return 2
            if l: return 1
            if ri: return 3
            return 2
        if ir: return 2
        if f: return 2
        if l: return 1
        if ri: return 3
        r=random.random()
        if r<0.65: return 2
        elif r<0.80: return 1
        elif r<0.95: return 3
        else: return random.choice([0,4])

    while ep_idx < n_total:
        buf = Buf(); ep_rews = []
        while buf.steps() < args.steps_per_iter and ep_idx < n_total:
            diff, wall = schedule[ep_idx]; ep_idx += 1
            env = OBELIX(scaling_factor=3, arena_size=500, max_steps=args.max_steps,
                         wall_obstacles=wall, difficulty=diff, box_speed=2,
                         seed=random.randint(0, 1000000))
            raw = env.sensor_feedback.copy()
            dr = DR(); tr = Tracker(); hx = None; pa = -1
            epr = 0; ps = False; pas = False; cs = 0; sd = 0; sn = 0

            frac = min(1.0, ep_idx / (n_total * args.teacher_anneal))
            pt = args.teacher_prob * (1 - frac)

            for s in range(args.max_steps):
                obs = build_obs(raw, pa, s, args.max_steps, dr, tr)
                if random.random() < pt:
                    a = t_act(raw, cs, sd, sn, tr.in_push)
                    _, lp, val, hx2 = model.get_action(obs, hx)
                else:
                    a, lp, val, hx2 = model.get_action(obs, hx)

                raw2, r, done = env.step(ACTION_LIST[a], render=False)
                sr = shape_reward(r, raw2, a, tr, dr, ps, pas)

                st = bool(raw2[17]); ans = any(raw2[j] for j in range(17))
                dr.update(a, st); tr.update(raw2)
                buf.add(obs, a, lp, sr, val, float(done))

                if st: cs += 1
                else: cs = 0
                if cs > 0 and cs % 9 == 0: sd = 1 - sd
                sn = 0 if ans else sn + 1
                ps = st; pas = ans; pa = a; raw = raw2.copy()
                epr += r; total_steps += 1; hx = hx2
                if done: break

            _, _, lv, _ = model.get_action(obs, hx)
            buf.end(lv); ep_rews.append(epr)

        o, a, lp, ret, adv = buf.compute(gamma=args.gamma, lam=args.gae_lambda)
        frac2 = min(1.0, total_steps / (n_total * args.max_steps * 0.7))
        ec = args.entropy_coef * (1-frac2) + 0.015 * frac2
        ppo_up(model, opt, o, a, lp, ret, adv, clip=args.clip, ec=ec,
               vc=args.value_coef, ep=args.n_epochs, bs=args.batch_size)
        buf.clear(); iteration += 1

        if iteration % 2 == 0:
            mr = np.mean(ep_rews) if ep_rews else 0
            print(f"Ep {ep_idx:5d}/{n_total} | Steps {total_steps:8d} | "
                  f"R={mr:8.1f} | ent={ec:.4f} | teacher={pt:.3f}")

        if iteration % args.eval_every == 0:
            for d in [0, 3]:
                for w in [False, True]:
                    er, es, esr = evaluate(model, d, w, n_eps=3)
                    print(f"  EVAL d={d} w={w}: R={er:.0f}±{es:.0f} SR={esr:.0%}")
            er0, _, _ = evaluate(model, 0, False, n_eps=5)
            if er0 > best:
                best = er0
                torch.save(model.state_dict(), args.save_path)
                print(f"  >> Saved best (R={er0:.0f})")

    torch.save(model.state_dict(), args.save_path.replace(".pth", "_final.pth"))
    print("\n" + "="*60 + "\nFinal Evaluation")
    for d in [0,2,3]:
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
    p.add_argument("--save_path", type=str, default="weights_ppo_v5.pth")
    args = p.parse_args()
    train(args)