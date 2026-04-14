"""
Tabular Q-Learning Training for OBELIX
=======================================
Based on the approach from:
  Mahadevan & Connell (1991/1992) - "Automatic Programming of 
  Behaviour-based Robots using Reinforcement Learning"

Key ideas:
1. Compress 18-bit observation → small integer state
2. Separate Q-tables for FIND mode vs PUSH mode
3. Eligibility traces (Q(lambda)) for long-horizon credit assignment
4. Smart exploration: epsilon-greedy BUT with forced FW bias in "no sensor" state
5. Curriculum: train level 0 first, then 2, then 3
6. The agent detects its own mode (find/push) from sensor history

WHY this works when PPO/DQN don't:
- The state space is SMALL (~50-200 reachable states) → tabular is exact
- No function approximation means no catastrophic forgetting
- Eligibility traces propagate the +2000 success reward back through 
  hundreds of steps, which PPO struggles with
- Forced forward exploration prevents the "always rotate" local optimum
"""

import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obelix import OBELIX
from tqdm import tqdm

ACTION_LIST = ["L45", "L22", "FW", "R22", "R45"]
NUM_ACTIONS = 5


# ──────────────────────────────────────────────
# State compression
# ──────────────────────────────────────────────
def compress_obs(obs):
    """
    Compress 18-bit observation into a meaningful state tuple.
    
    Sensor layout (from code analysis):
      [0,1]   = left-rear sonar (far, near)
      [2,3]   = left-front sonar (far, near)
      [4,5]   = front-left-wide sonar (far, near)
      [6,7]   = front-left-narrow sonar (far, near)
      [8,9]   = front-right-narrow sonar (far, near)
      [10,11] = front-right-wide sonar (far, near)
      [12,13] = right-front sonar (far, near)
      [14,15] = right-rear sonar (far, near)
      [16]    = IR (front, very short range)
      [17]    = stuck flag
    
    Compress to: (left_region, front_region, right_region, ir, stuck)
    Where each region is: 0=nothing, 1=far, 2=near, 3=both
    """
    o = np.array(obs, dtype=int)
    
    # Left region: combine sensors 0-3 (left-rear + left-front)
    left_far = int(o[0] or o[2])
    left_near = int(o[1] or o[3])
    left = left_far + 2 * left_near  # 0,1,2,3
    
    # Front region: combine sensors 4-11 (all 4 forward-facing sonars)
    front_far = int(o[4] or o[6] or o[8] or o[10])
    front_near = int(o[5] or o[7] or o[9] or o[11])
    front = front_far + 2 * front_near  # 0,1,2,3
    
    # Right region: combine sensors 12-15 (right-front + right-rear)
    right_far = int(o[12] or o[14])
    right_near = int(o[13] or o[15])
    right = right_far + 2 * right_near  # 0,1,2,3
    
    ir = int(o[16])
    stuck = int(o[17])
    
    return (left, front, right, ir, stuck)

# Total possible compressed states: 4 * 4 * 4 * 2 * 2 = 256
# But many are unreachable, so actual count is much smaller


def compress_obs_fine(obs):
    """
    Finer compression that preserves left/right asymmetry better.
    Keeps front-left vs front-right distinction.
    
    Returns: (left, front_left, front_right, right, ir, stuck)
    Each sub-region: 0=nothing, 1=far, 2=near, 3=both
    """
    o = np.array(obs, dtype=int)
    
    left_far = int(o[0] or o[2])
    left_near = int(o[1] or o[3])
    left = left_far + 2 * left_near
    
    fl_far = int(o[4] or o[6])
    fl_near = int(o[5] or o[7])
    front_left = fl_far + 2 * fl_near
    
    fr_far = int(o[8] or o[10])
    fr_near = int(o[9] or o[11])
    front_right = fr_far + 2 * fr_near
    
    right_far = int(o[12] or o[14])
    right_near = int(o[13] or o[15])
    right = right_far + 2 * right_near
    
    ir = int(o[16])
    stuck = int(o[17])
    
    return (left, front_left, front_right, right, ir, stuck)

# Total: 4^4 * 2 * 2 = 1024 max, but many unreachable


# ──────────────────────────────────────────────
# Mode detection from observation history
# ──────────────────────────────────────────────
class ModeDetector:
    """
    Detect whether we're in FIND or PUSH mode.
    Push mode: IR sensor was active AND we moved forward AND 
    sensors keep firing consistently.
    
    The environment sets enable_push internally but doesn't expose it.
    We infer it from: IR fires → next step still has sensors → we're attached.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.in_push = False
        self.ir_was_on = False
        self.consecutive_sensor_steps = 0
        self.step = 0
    
    def update(self, obs, action_idx):
        self.step += 1
        ir = bool(obs[16])
        any_near = any(obs[2*i+1] for i in range(8))
        any_sensor = any(obs[j] for j in range(17))
        
        if not self.in_push:
            # Transition to push: IR was on, we went forward, and sensors still active
            if self.ir_was_on and any_sensor:
                self.consecutive_sensor_steps += 1
                if self.consecutive_sensor_steps >= 2:
                    self.in_push = True
            elif any_sensor:
                self.consecutive_sensor_steps += 1
            else:
                self.consecutive_sensor_steps = 0
            
            # Also detect push if IR fires AND near sensors fire simultaneously
            # This means bot body overlaps box
            if ir and any_near:
                self.in_push = True
        
        self.ir_was_on = ir
    
    def get_mode(self):
        return "push" if self.in_push else "find"


# ──────────────────────────────────────────────
# Q-Learning Agent with Eligibility Traces
# ──────────────────────────────────────────────
class QLambdaAgent:
    def __init__(self, alpha=0.1, gamma=0.995, lambd=0.9, 
                 epsilon=0.3, epsilon_min=0.05, epsilon_decay=0.9999,
                 use_fine_compression=True):
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_fine = use_fine_compression
        
        # Separate Q-tables for find and push modes
        self.q_find = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.q_push = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        
        # Eligibility traces
        self.e_find = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.e_push = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        
        self.mode_detector = ModeDetector()
        self.prev_state = None
        self.prev_action = None
        self.prev_mode = None
        
        # Exploration tracking
        self.episode_count = 0
        self.total_steps = 0
        
        # Stats
        self.state_visit_counts = defaultdict(int)
    
    def compress(self, obs):
        if self.use_fine:
            return compress_obs_fine(obs)
        else:
            return compress_obs(obs)
    
    def get_q_table(self, mode):
        return self.q_find if mode == "find" else self.q_push
    
    def get_e_table(self, mode):
        return self.e_find if mode == "find" else self.e_push
    
    def select_action(self, obs, mode, training=True):
        """
        Epsilon-greedy with SMART exploration:
        - In "no sensor" state: force FW with high probability
        - In sensor states: standard epsilon-greedy
        """
        state = self.compress(obs)
        q = self.get_q_table(mode)
        stuck = bool(obs[17])
        any_sensor = any(obs[j] for j in range(17))
        
        if training and random.random() < self.epsilon:
            # Smart exploration, not uniform random
            if stuck:
                # When stuck: always turn (50/50 left/right, prefer bigger turn)
                return random.choice([0, 4])  # L45 or R45
            elif not any_sensor:
                # No sensors: HEAVILY bias toward forward
                # This is THE critical trick. Random walk = forward mostly + occasional turns
                r = random.random()
                if r < 0.6:
                    return 2  # FW
                elif r < 0.75:
                    return 1  # L22
                elif r < 0.90:
                    return 3  # R22
                elif r < 0.95:
                    return 0  # L45
                else:
                    return 4  # R45
            else:
                # Sensors active: explore all actions but bias toward greedy
                if random.random() < 0.3:
                    return random.randint(0, NUM_ACTIONS - 1)
                else:
                    return int(np.argmax(q[state]))
        else:
            # Greedy with tie-breaking
            q_vals = q[state]
            max_q = np.max(q_vals)
            max_actions = np.where(np.abs(q_vals - max_q) < 1e-8)[0]
            
            if len(max_actions) > 1:
                # Tie-breaking: prefer FW, then small turns, then big turns
                priority = {2: 0, 1: 1, 3: 1, 0: 2, 4: 2}
                best = min(max_actions, key=lambda a: priority.get(a, 3))
                return int(best)
            return int(max_actions[0])
    
    def begin_episode(self):
        """Call at start of each episode."""
        self.mode_detector.reset()
        self.prev_state = None
        self.prev_action = None
        self.prev_mode = None
        # Clear eligibility traces
        for key in self.e_find:
            self.e_find[key][:] = 0
        for key in self.e_push:
            self.e_push[key][:] = 0
        self.episode_count += 1
    
    def step(self, obs, reward, done, training=True):
        """
        One step of Q(lambda) learning.
        Returns the action to take.
        """
        # Update mode
        if self.prev_action is not None:
            self.mode_detector.update(obs, self.prev_action)
        mode = self.mode_detector.get_mode()
        
        state = self.compress(obs)
        self.state_visit_counts[state] += 1
        self.total_steps += 1
        
        # Select action
        action = self.select_action(obs, mode, training=training)
        
        # Learning update (if we have a previous transition)
        if training and self.prev_state is not None:
            q = self.get_q_table(self.prev_mode)
            e = self.get_e_table(self.prev_mode)
            
            # TD error
            if done:
                td_target = reward
            else:
                q_current = self.get_q_table(mode)
                td_target = reward + self.gamma * np.max(q_current[state])
            
            td_error = td_target - q[self.prev_state][self.prev_action]
            
            # Update eligibility trace (replacing traces)
            e[self.prev_state][:] = 0
            e[self.prev_state][self.prev_action] = 1.0
            
            # Update all states with non-zero traces
            for s in list(e.keys()):
                if np.any(np.abs(e[s]) > 1e-10):
                    q[s] += self.alpha * td_error * e[s]
                    e[s] *= self.gamma * self.lambd
                    # Clean up tiny traces
                    if np.all(np.abs(e[s]) < 1e-10):
                        del e[s]
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store for next step
        self.prev_state = state
        self.prev_action = action
        self.prev_mode = mode
        
        return action
    
    def save(self, path):
        """Save Q-tables and params."""
        data = {
            "q_find": {str(k): v.tolist() for k, v in self.q_find.items()},
            "q_push": {str(k): v.tolist() for k, v in self.q_push.items()},
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "use_fine": self.use_fine,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambd": self.lambd,
            "state_visits": {str(k): v for k, v in self.state_visit_counts.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved Q-tables to {path} "
              f"(find: {len(self.q_find)} states, push: {len(self.q_push)} states)")
    
    def load(self, path):
        """Load Q-tables."""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.q_find = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.q_push = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        
        for k, v in data["q_find"].items():
            self.q_find[eval(k)] = np.array(v)
        for k, v in data["q_push"].items():
            self.q_push[eval(k)] = np.array(v)
        
        self.epsilon = data.get("epsilon", 0.05)
        self.episode_count = data.get("episode_count", 0)
        self.total_steps = data.get("total_steps", 0)
        self.use_fine = data.get("use_fine", True)
        
        print(f"Loaded Q-tables from {path} "
              f"(find: {len(self.q_find)} states, push: {len(self.q_push)} states)")


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate(agent, difficulty=0, wall=False, n_episodes=10, max_steps=2000, verbose=False):
    rewards = []
    successes = []
    steps_list = []
    
    for ep in range(n_episodes):
        env = OBELIX(
            scaling_factor=3, arena_size=500, max_steps=max_steps,
            wall_obstacles=wall, difficulty=difficulty, box_speed=2,
            seed=ep * 1000 + 42,
        )
        raw_obs = env.sensor_feedback.copy()
        agent.begin_episode()
        
        total_reward = 0
        for step in range(max_steps):
            action_idx = agent.step(raw_obs, 0, False, training=False)
            action = ACTION_LIST[action_idx]
            obs, reward, done = env.step(action, render=False)
            total_reward += reward
            raw_obs = obs.copy()
            if done:
                break
        
        rewards.append(total_reward)
        successes.append(1 if (env.enable_push and env.done) else 0)
        steps_list.append(step + 1)
        
        if verbose:
            print(f"  Ep {ep}: R={total_reward:.0f}, Steps={step+1}, "
                  f"Success={'YES' if successes[-1] else 'no'}, "
                  f"Push={env.enable_push}")
    
    return np.mean(rewards), np.std(rewards), np.mean(successes), np.mean(steps_list)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train(args):
    print("=" * 60)
    print("Q(lambda) Training for OBELIX")
    print("=" * 60)
    
    agent = QLambdaAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        lambd=args.lambd,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        use_fine_compression=True,
    )
    
    best_eval_reward = -float("inf")
    all_rewards = []
    all_eval_rewards = []
    
    # Curriculum: level 0 → level 0+wall → level 2 → level 2+wall → level 3 → level 3+wall
    curriculum = [
        (0, False, args.episodes_per_level),
        (0, True, args.episodes_per_level // 2),
        (2, False, args.episodes_per_level),
        (2, True, args.episodes_per_level // 2),
        (3, False, args.episodes_per_level),
        (3, True, args.episodes_per_level // 2),
    ]
    
    if not args.curriculum:
        curriculum = [(args.difficulty, args.wall_obstacles, args.n_episodes)]
    
    global_ep = 0
    
    for diff, wall, n_eps in curriculum:
        print(f"\n--- Training: difficulty={diff}, wall={wall}, episodes={n_eps} ---")
        
        for ep in tqdm(range(n_eps)):
            seed = random.randint(0, 1_000_000)
            env = OBELIX(
                scaling_factor=3, arena_size=500, max_steps=args.max_steps,
                wall_obstacles=wall, difficulty=diff, box_speed=2, seed=seed,
            )
            raw_obs = env.sensor_feedback.copy()
            agent.begin_episode()
            
            total_reward = 0
            prev_reward = 0
            
            for step in range(args.max_steps):
                action_idx = agent.step(raw_obs, prev_reward, False, training=True)
                action = ACTION_LIST[action_idx]
                obs, reward, done = env.step(action, render=False)
                
                total_reward += reward
                prev_reward = reward
                raw_obs = obs.copy()
                
                if done:
                    # Final update with terminal reward
                    agent.step(obs, reward, True, training=True)
                    break
            
            all_rewards.append(total_reward)
            global_ep += 1
            
            # Logging
            if (ep + 1) % args.log_interval == 0:
                recent = all_rewards[-args.log_interval:]
                print(f"  Ep {global_ep:5d} | MeanR(last{args.log_interval})={np.mean(recent):8.1f} | "
                      f"eps={agent.epsilon:.4f} | "
                      f"States: find={len(agent.q_find)}, push={len(agent.q_push)}")
            
            # Evaluate periodically
            if (ep + 1) % args.eval_interval == 0:
                eval_r, eval_s, eval_sr, eval_st = evaluate(
                    agent, diff, wall, n_episodes=5, verbose=False
                )
                all_eval_rewards.append(eval_r)
                print(f"  >> EVAL (d={diff},w={wall}): R={eval_r:.1f}±{eval_s:.1f}, "
                      f"Success={eval_sr:.0%}, Steps={eval_st:.0f}")
                
                if eval_r > best_eval_reward:
                    best_eval_reward = eval_r
                    agent.save(args.save_path)
                    print(f"  >> Saved best model (R={eval_r:.1f})")
    
    # Final save
    agent.save(args.save_path.replace(".json", "_final.json"))
    
    # Final evaluation across all levels
    print("\n" + "=" * 60)
    print("Final Evaluation (10 episodes each):")
    print("=" * 60)
    for diff in [0, 2, 3]:
        for wall in [False, True]:
            r, s, sr, st = evaluate(agent, diff, wall, n_episodes=10, verbose=False)
            print(f"  Diff={diff} Wall={str(wall):5s}: "
                  f"R={r:8.1f}±{s:6.1f}, Success={sr:.0%}, Steps={st:.0f}")
    
    # Save training log
    np.savez(
        args.save_path.replace(".json", "_log.npz"),
        ep_rewards=np.array(all_rewards),
        eval_rewards=np.array(all_eval_rewards),
    )
    
    # Print Q-table insights
    print(f"\nQ-table sizes: find={len(agent.q_find)}, push={len(agent.q_push)}")
    print(f"Total unique states visited: {len(agent.state_visit_counts)}")
    
    # Show top states and their Q-values
    print("\nTop FIND mode Q-values:")
    for state in sorted(agent.q_find.keys(), 
                        key=lambda s: np.max(agent.q_find[s]), reverse=True)[:10]:
        q = agent.q_find[state]
        best = ACTION_LIST[np.argmax(q)]
        print(f"  {state} → {best} (Q={q})")
    
    print("\nTop PUSH mode Q-values:")
    for state in sorted(agent.q_push.keys(),
                        key=lambda s: np.max(agent.q_push[s]), reverse=True)[:10]:
        q = agent.q_push[state]
        best = ACTION_LIST[np.argmax(q)]
        print(f"  {state} → {best} (Q={q})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.15, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--lambd", type=float, default=0.85, help="Eligibility trace decay")
    parser.add_argument("--epsilon", type=float, default=0.4, help="Initial exploration")
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.99995)
    parser.add_argument("--n_episodes", type=int, default=5000)
    parser.add_argument("--episodes_per_level", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--curriculum", action="store_true", default=True)
    parser.add_argument("--no_curriculum", dest="curriculum", action="store_false")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="weights_ql_trace.json")
    args = parser.parse_args()
    train(args)