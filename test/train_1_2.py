"""
Tabular Q(lambda) Training for OBELIX — v2
============================================
Fixes from v1 diagnosis:
  A. Wall-awareness: "stuck_after_sensor" flag in state tells agent 
     "what you're sensing is probably a wall, not the box"
  B. Better exploration: pseudo-random turns using obs hash, not just step count.
     Also: escalating turn probability when stuck repeatedly.
  C. Separate training phases with heavy wall emphasis
  D. "Stale sensor" counter: how many consecutive steps sensors have been on
     without attachment → high count = probably wall, not box
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from collections import defaultdict

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obelix_fast import OBELIXFast as OBELIX


ACTION_LIST = ["L45", "L22", "FW", "R22", "R45"]
NUM_ACTIONS = 5


def compress_state(obs, wall_suspect, stale_sensor, in_push):
    """
    State = (left, front_left, front_right, right, ir, stuck, wall_suspect, stale_sensor_bin, in_push)
    
    wall_suspect: 0 or 1 — set to 1 when we hit stuck=1 right after sensors were on
                  This means "the thing you're detecting is a wall, not the box"
    
    stale_sensor_bin: 0,1,2 — how many consecutive steps with sensors on but no attachment
                      0 = fresh (0-3 steps), 1 = stale (4-15), 2 = very stale (16+)
                      High staleness + no IR = probably wall
    
    in_push: 0 or 1 — whether we believe we've attached to the box
    """
    o = np.asarray(obs, dtype=int)
    
    left = int(o[0] or o[1] or o[2] or o[3])       # any left sensor
    fl = int(o[4] or o[5] or o[6] or o[7])          # any front-left
    fr = int(o[8] or o[9] or o[10] or o[11])        # any front-right
    right = int(o[12] or o[13] or o[14] or o[15])   # any right sensor
    
    # Near vs far for front (important for approach)
    front_near = int(o[5] or o[7] or o[9] or o[11])
    
    ir = int(o[16])
    stuck = int(o[17])
    
    # Bin stale_sensor: 0=fresh, 1=stale, 2=very_stale
    if stale_sensor <= 3:
        ss_bin = 0
    elif stale_sensor <= 15:
        ss_bin = 1
    else:
        ss_bin = 2
    
    return (left, fl, fr, right, front_near, ir, stuck, int(wall_suspect), ss_bin, int(in_push))


class ModeDetector:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.in_push = False
        self.ir_was_on = False
        self.consec = 0
    
    def update(self, obs):
        ir = bool(obs[16])
        any_near = any(bool(obs[2*i+1]) for i in range(8))
        any_s = any(bool(obs[j]) for j in range(17))
        if not self.in_push:
            if self.ir_was_on and any_s:
                self.consec += 1
                if self.consec >= 2:
                    self.in_push = True
            elif any_s:
                self.consec += 1
            else:
                self.consec = 0
            if ir and any_near:
                self.in_push = True
        self.ir_was_on = ir
        return self.in_push


class QLambdaAgent:
    def __init__(self, alpha=0.1, gamma=0.995, lambd=0.85,
                 epsilon=0.4, epsilon_min=0.05, epsilon_decay=0.99995):
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.e = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        
        self.mode = ModeDetector()
        self.prev_state = None
        self.prev_action = None
        
        # Tracking for wall detection
        self.wall_suspect = False
        self.stale_sensor = 0       # consecutive steps with sensors on, no attachment
        self.prev_any_sensor = False
        self.prev_action_was_fw = False
        self.stuck_history = []      # recent stuck events for exploration
        self.steps_no_sensor = 0     # for exploration escalation
        
        self.episode_count = 0
        self.total_steps = 0
        self.state_visits = defaultdict(int)
        self._consec_stuck = 0
        self._stuck_dir = 0
    
    def begin_episode(self):
        self.mode.reset()
        self.prev_state = None
        self.prev_action = None
        self.wall_suspect = False
        self.stale_sensor = 0
        self.prev_any_sensor = False
        self.prev_action_was_fw = False
        self.stuck_history = []
        self.steps_no_sensor = 0
        self._consec_stuck = 0
        self._stuck_dir = 0
        for key in list(self.e.keys()):
            del self.e[key]
        self.episode_count += 1
    
    def _update_wall_tracking(self, obs):
        """Track whether current sensor readings are likely a wall."""
        stuck = bool(obs[17])
        any_sensor = any(bool(obs[j]) for j in range(17))
        ir = bool(obs[16])
        
        # Wall suspect: we had sensors on, went forward, got stuck
        if stuck and self.prev_any_sensor and self.prev_action_was_fw:
            self.wall_suspect = True
            self.stuck_history.append(1)
        elif stuck:
            self.stuck_history.append(1)
        else:
            if len(self.stuck_history) > 0 and self.stuck_history[-1] == 1:
                self.stuck_history.append(0)
        
        # Keep only last 10
        self.stuck_history = self.stuck_history[-10:]
        
        # Clear wall suspect if we turn away and sensors go off
        if not any_sensor and not stuck:
            self.wall_suspect = False
            self.stale_sensor = 0
        
        # Clear wall suspect if IR fires (IR = very close = probably box, not wall)
        # Actually wall also triggers IR if close enough... but box is smaller
        # Keep wall_suspect if stuck count is high
        
        # Stale sensor: sensors on but not attaching
        if any_sensor and not self.mode.in_push:
            self.stale_sensor += 1
        else:
            if not any_sensor:
                self.stale_sensor = 0
        
        # Steps without any sensor (for exploration escalation)
        if any_sensor:
            self.steps_no_sensor = 0
        else:
            self.steps_no_sensor += 1
        
        self.prev_any_sensor = any_sensor
    
    def select_action(self, obs, training=True):
        in_push = self.mode.in_push
        state = compress_state(obs, self.wall_suspect, self.stale_sensor, in_push)
        self.state_visits[state] += 1
        
        stuck = bool(obs[17])
        any_sensor = any(bool(obs[j]) for j in range(17))
        ir = bool(obs[16])
        
        if training and random.random() < self.epsilon:
            action = self._smart_explore(obs, in_push, stuck, any_sensor, ir)
        elif not any_sensor and not stuck and not in_push:
            # NO SENSORS, NOT STUCK, NOT PUSHING:
            # Q-table has no useful signal here (all actions get -1).
            # Use random walk exploration even in greedy mode.
            # This is the correct approach — random walk IS the optimal
            # search strategy in a bounded arena with no information.
            action = self._smart_explore(obs, in_push, stuck, any_sensor, ir)
        else:
            q_vals = self.q[state]
            max_q = np.max(q_vals)
            max_actions = np.where(np.abs(q_vals - max_q) < 1e-8)[0]
            
            if len(max_actions) > 1:
                # Tie-breaking
                if in_push:
                    priority = {2:0, 1:1, 3:1, 0:2, 4:2}
                else:
                    priority = {2:0, 1:1, 3:1, 0:2, 4:2}
                action = int(min(max_actions, key=lambda a: priority.get(a,3)))
            else:
                action = int(max_actions[0])
            
            # Safety override: if stuck, use turn-turn-FW cycle
            # stuck_flag ONLY clears on successful FW, so we MUST try FW periodically
            if stuck and action == 2:
                # Check if we should try FW (every 3rd stuck step) or turn
                if self._consec_stuck % 3 < 2:
                    # Turn phase
                    q_no_fw = q_vals.copy()
                    q_no_fw[2] = -1e9
                    action = int(np.argmax(q_no_fw))
                # else: keep FW to test if we're free
            
            # Safety override: if wall suspected and heading toward it, turn away
            if self.wall_suspect and not in_push and action == 2 and self.stale_sensor > 5:
                # High stale sensor + wall suspect = definitely wall, turn away
                q_no_fw = q_vals.copy()
                q_no_fw[2] = -1e9
                action = int(np.argmax(q_no_fw))
        
        self.prev_action_was_fw = (action == 2)
        return state, action
    
    def _smart_explore(self, obs, in_push, stuck, any_sensor, ir):
        """Context-aware exploration."""
        # STUCK: turn-turn-FW cycle (FW is REQUIRED to clear stuck_flag)
        if stuck:
            cycle = self._consec_stuck % 3
            if cycle < 2:
                # Turn phase
                if self._consec_stuck > 9:
                    self._stuck_dir = 1 - getattr(self, '_stuck_dir', 0)
                if getattr(self, '_stuck_dir', 0) == 0:
                    return random.choice([0, 1])  # left turns
                else:
                    return random.choice([3, 4])  # right turns
            else:
                return 2  # FW — test if we're free
        
        # PUSH MODE: mostly forward
        if in_push:
            if ir:
                return 2  # FW — pushing
            r = random.random()
            if r < 0.7: return 2    # FW
            elif r < 0.85: return 1  # L22
            else: return 3           # R22
        
        # WALL SUSPECTED: turn away, don't go forward
        if self.wall_suspect:
            return random.choice([0, 1, 3, 4])  # any turn
        
        # STALE SENSOR (sensors on for many steps, not attaching): probably wall
        if any_sensor and self.stale_sensor > 10:
            # Turn away from what we're sensing
            o = np.asarray(obs, dtype=int)
            left_any = bool(o[0] or o[1] or o[2] or o[3])
            right_any = bool(o[12] or o[13] or o[14] or o[15])
            if left_any and not right_any:
                return random.choice([3, 4])  # turn right
            elif right_any and not left_any:
                return random.choice([0, 1])  # turn left
            else:
                return random.choice([0, 4])  # big turn either way
        
        # IR ACTIVE: go forward (close to box)
        if ir:
            return 2
        
        # SENSORS ACTIVE: bias toward box
        if any_sensor:
            o = np.asarray(obs, dtype=int)
            left_any = bool(o[0] or o[1] or o[2] or o[3])
            front_any = bool(o[4] or o[5] or o[6] or o[7] or o[8] or o[9] or o[10] or o[11])
            right_any = bool(o[12] or o[13] or o[14] or o[15])
            
            if front_any:
                r = random.random()
                if r < 0.7: return 2    # FW toward box
                elif r < 0.85: return 1  # slight adjust
                else: return 3
            elif left_any:
                return random.choice([0, 1])  # turn left
            elif right_any:
                return random.choice([3, 4])  # turn right
            else:
                return 2  # forward
        
        # NO SENSORS: explore with pseudo-random walk
        # Use hash of recent history for variety
        h = hash(tuple(self.stuck_history[-3:])) ^ self.total_steps
        
        # Escalate turning based on how long since last sensor
        if self.steps_no_sensor < 30:
            # Early: mostly forward (searching)
            r = (h % 100) / 100.0
            if r < 0.65: return 2    # FW
            elif r < 0.80: return 1  # L22
            elif r < 0.95: return 3  # R22
            else: return random.choice([0, 4])
        elif self.steps_no_sensor < 80:
            # Medium: add more turns to sweep
            r = (h % 100) / 100.0
            if r < 0.45: return 2    # FW
            elif r < 0.65: return 1  # L22
            elif r < 0.85: return 3  # R22
            else: return random.choice([0, 4])
        else:
            # Long time without sensor: big turns to change direction
            r = (h % 100) / 100.0
            if r < 0.30: return 2
            elif r < 0.50: return 0  # L45
            elif r < 0.70: return 4  # R45
            elif r < 0.85: return 1
            else: return 3
    
    def learn(self, obs, reward, done):
        """Q(lambda) update."""
        if self.prev_state is None:
            return
        
        in_push = self.mode.in_push
        state = compress_state(obs, self.wall_suspect, self.stale_sensor, in_push)
        
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q[state])
        
        td_error = td_target - self.q[self.prev_state][self.prev_action]
        
        # Replacing traces
        self.e[self.prev_state][:] = 0
        self.e[self.prev_state][self.prev_action] = 1.0
        
        for s in list(self.e.keys()):
            trace = self.e[s]
            if np.any(np.abs(trace) > 1e-10):
                self.q[s] += self.alpha * td_error * trace
                trace *= self.gamma * self.lambd
                if np.all(np.abs(trace) < 1e-10):
                    del self.e[s]
            else:
                del self.e[s]
    
    def step(self, obs, reward, done, training=True):
        self.total_steps += 1
        self.mode.update(obs)
        self._update_wall_tracking(obs)
        
        # Track consecutive stuck steps for turn-turn-FW cycle
        if bool(obs[17]):
            self._consec_stuck += 1
        else:
            self._consec_stuck = 0
        
        state, action = self.select_action(obs, training=training)
        
        if training:
            self.learn(obs, reward, done)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def save(self, path):
        data = {
            "q": {str(k): v.tolist() for k, v in self.q.items()},
            "epsilon": self.epsilon,
            "episodes": self.episode_count,
            "steps": self.total_steps,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved: {len(self.q)} states → {path}")
    
    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        for k, v in data.get("q", {}).items():
            self.q[eval(k)] = np.array(v)
        self.epsilon = data.get("epsilon", 0.05)
        print(f"Loaded: {len(self.q)} states from {path}")


def evaluate(agent, difficulty=0, wall=False, n_eps=10, max_steps=2000, verbose=False):
    rewards, successes, steps_list = [], [], []
    for ep in range(n_eps):
        env = OBELIX(scaling_factor=3, arena_size=500, max_steps=max_steps,
                     wall_obstacles=wall, difficulty=difficulty, box_speed=2,
                     seed=ep*1000+42)
        obs = env.sensor_feedback.copy()
        agent.begin_episode()
        total_r, prev_r = 0, 0
        for step in range(max_steps):
            a = agent.step(obs, prev_r, False, training=False)
            obs, r, done = env.step(ACTION_LIST[a], render=False)
            total_r += r; prev_r = r
            if done:
                agent.step(obs, r, True, training=False)
                break
        rewards.append(total_r)
        successes.append(1 if env.enable_push and env.done else 0)
        steps_list.append(step+1)
        if verbose:
            print(f"  ep{ep}: R={total_r:.0f} steps={step+1} "
                  f"{'SUCCESS' if successes[-1] else 'fail'} push={env.enable_push}")
    return np.mean(rewards), np.std(rewards), np.mean(successes), np.mean(steps_list)


def train(args):
    print("=" * 60)
    print("Q(lambda) Training v2 — Wall-Aware + Better Exploration")
    print("=" * 60)
    
    agent = QLambdaAgent(
        alpha=args.alpha, gamma=args.gamma, lambd=args.lambd,
        epsilon=args.epsilon, epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )
    
    if args.resume and os.path.exists(args.save_path):
        agent.load(args.save_path)
    
    best_eval = -float("inf")
    all_rewards = []
    all_eval = []
    
    # Training schedule: alternate between wall and no-wall heavily
    # More wall episodes because that's where the agent fails
    schedule = []
    if args.curriculum:
        for diff in [0, 2, 3]:
            n = args.episodes_per_level
            # Interleave wall and no-wall to learn both
            for batch in range(n // 20):
                schedule.extend([(diff, False)] * 10)
                schedule.extend([(diff, True)] * 10)
        # Extra wall-only training at the end
        for diff in [0, 2, 3]:
            schedule.extend([(diff, True)] * (args.episodes_per_level // 3))
    else:
        schedule = [(args.difficulty, args.wall_obstacles)] * args.n_episodes
    
    for ep_idx, (diff, wall) in tqdm(enumerate(schedule), total=len(schedule)):
        seed = random.randint(0, 1_000_000)
        env = OBELIX(scaling_factor=3, arena_size=500, max_steps=args.max_steps,
                     wall_obstacles=wall, difficulty=diff, box_speed=2, seed=seed)
        obs = env.sensor_feedback.copy()
        agent.begin_episode()
        total_r, prev_r = 0, 0
        
        for step in range(args.max_steps):
            a = agent.step(obs, prev_r, False, training=True)
            obs, r, done = env.step(ACTION_LIST[a], render=False)
            total_r += r; prev_r = r
            if done:
                agent.step(obs, r, True, training=True)
                break
        
        all_rewards.append(total_r)
        
        if (ep_idx+1) % args.log_interval == 0:
            recent = all_rewards[-args.log_interval:]
            print(f"Ep {ep_idx+1:5d}/{len(schedule)} | "
                  f"R={np.mean(recent):8.1f} | eps={agent.epsilon:.4f} | "
                  f"d={diff} w={wall} | Q-states={len(agent.q)}")
        
        if (ep_idx+1) % args.eval_interval == 0:
            # Quick eval on current level
            er, es, esr, est = evaluate(agent, diff, wall, n_eps=5, verbose=False)
            all_eval.append(er)
            print(f"  >> EVAL d={diff} w={wall}: R={er:.1f}±{es:.1f} SR={esr:.0%} steps={est:.0f}")
            
            if er > best_eval:
                best_eval = er
                agent.save(args.save_path)
    
    # Final save
    agent.save(args.save_path.replace(".json", "_final.json"))
    np.savez(args.save_path.replace(".json", "_log.npz"),
             ep_rewards=np.array(all_rewards), eval_rewards=np.array(all_eval))
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (10 episodes each)")
    print("=" * 60)
    total_mean = 0
    for diff in [0, 2, 3]:
        for wall in [False, True]:
            r, s, sr, st = evaluate(agent, diff, wall, n_eps=10, verbose=False)
            print(f"  d={diff} w={str(wall):5s}: R={r:8.1f}±{s:6.1f} SR={sr:.0%} steps={st:.0f}")
            total_mean += r
    print(f"\n  Average across all: {total_mean/6:.1f}")
    
    print(f"\nQ-table size: {len(agent.q)} states")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lambd", type=float, default=0.85)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.99999)
    parser.add_argument("--n_episodes", type=int, default=5000)
    parser.add_argument("--episodes_per_level", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--curriculum", action="store_true", default=True)
    parser.add_argument("--no_curriculum", dest="curriculum", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="weights.json")
    args = parser.parse_args()
    train(args)