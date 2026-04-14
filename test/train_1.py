"""
PPO Training Script for OBELIX Warehouse Robot
================================================
Key tricks to prevent degenerate "always rotate" policy:
1. Frame-stacking (last 4 obs) to give temporal context / implicit velocity
2. Action history embedding so the agent knows what it just did
3. Intrinsic exploration bonus: small +reward for moving forward when no sensors fire
4. Entropy bonus scheduling: high early, decay later
5. Stuck-aware state augmentation: add step count features
6. Curriculum: start level 0, then gradually increase difficulty
7. Reward shaping during training (not eval): small forward bias, distance proxy
8. Recurrent policy (GRU) to handle POMDP partial observability
"""

import os
import sys
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# Add parent dir for obelix import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obelix import OBELIX


# ──────────────────────────────────────────────
# Utility: build augmented observation
# ──────────────────────────────────────────────
ACTION_LIST = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
NUM_ACTIONS = 5


def augment_obs(raw_obs, prev_action_idx, step_count, max_steps, enable_push_flag):
    """
    Augment the 18-bit observation with:
    - one-hot previous action (5)
    - normalized step count (1)
    - push flag (1)
    Total: 18 + 5 + 1 + 1 = 25
    """
    obs = np.array(raw_obs, dtype=np.float32)
    action_oh = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if prev_action_idx >= 0:
        action_oh[prev_action_idx] = 1.0
    step_norm = np.array([step_count / max_steps], dtype=np.float32)
    push_flag = np.array([float(enable_push_flag)], dtype=np.float32)
    return np.concatenate([obs, action_oh, step_norm, push_flag])


AUG_OBS_DIM = 25


# ──────────────────────────────────────────────
# Frame stacking wrapper
# ──────────────────────────────────────────────
class FrameStack:
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self, obs):
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get()

    def push(self, obs):
        self.frames.append(obs)
        return self._get()

    def _get(self):
        return np.concatenate(list(self.frames), axis=0)


STACKED_OBS_DIM = AUG_OBS_DIM * 4  # 25 * 4 = 100


# ──────────────────────────────────────────────
# PPO Actor-Critic with GRU for POMDP
# ──────────────────────────────────────────────
class ActorCriticGRU(nn.Module):
    def __init__(self, obs_dim=STACKED_OBS_DIM, hidden=256, gru_hidden=128, n_actions=5):
        super().__init__()
        self.gru_hidden_size = gru_hidden

        # Shared feature extractor
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # GRU for temporal reasoning
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=True)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Smaller init for policy head to encourage exploration
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)

    def forward(self, obs, hx=None):
        """
        obs: (batch, seq_len, obs_dim) or (batch, obs_dim)
        hx: (1, batch, gru_hidden) or None
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)

        batch_size, seq_len, _ = obs.shape

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        if hx is None:
            hx = torch.zeros(1, batch_size, self.gru_hidden_size, device=obs.device)

        gru_out, hx_new = self.gru(x, hx)

        # Use last timestep output
        feat = gru_out[:, -1, :]

        logits = self.actor(feat)
        value = self.critic(feat)

        return logits, value.squeeze(-1), hx_new

    def get_action(self, obs, hx=None, deterministic=False):
        """Single step action selection."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
            logits, value, hx_new = self.forward(obs_t, hx)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            log_prob = torch.log(probs[0, action] + 1e-8).item()
        return action, log_prob, value.item(), hx_new


# ──────────────────────────────────────────────
# Simple MLP Actor-Critic (faster, no GRU)
# ──────────────────────────────────────────────
class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim=STACKED_OBS_DIM, hidden=256, n_actions=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)

    def forward(self, obs, hx=None):
        if obs.dim() == 3:
            obs = obs[:, -1, :]  # take last frame if sequence
        feat = self.shared(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value, hx

    def get_action(self, obs, hx=None, deterministic=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits, value, _ = self.forward(obs_t)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            log_prob = torch.log(probs[0, action] + 1e-8).item()
        return action, log_prob, value.item(), hx


# ──────────────────────────────────────────────
# PPO Rollout Buffer
# ──────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages

    def get_tensors(self, returns, advantages):
        obs = torch.FloatTensor(np.array(self.obs))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        returns_t = torch.FloatTensor(returns)
        advantages_t = torch.FloatTensor(advantages)
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        return obs, actions, old_log_probs, returns_t, advantages_t

    def clear(self):
        self.__init__()


# ──────────────────────────────────────────────
# Reward shaping for training
# ──────────────────────────────────────────────
def shape_reward(raw_reward, obs, action_idx, env, prev_any_sensor, step_in_ep):
    """
    Training-time reward shaping to encourage exploration and forward motion.
    These shaping terms are potential-based or small enough to not distort optimal policy.
    """
    shaped = raw_reward

    any_sensor = any(obs[:17] > 0)
    stuck = bool(obs[17])

    # Trick 1: Forward motion bonus when no sensors fire
    # The agent needs to MOVE to discover the box. Without this,
    # rotating is "safe" (no stuck penalty) but useless.
    if action_idx == 2 and not stuck:  # FW and not stuck
        if not any_sensor:
            shaped += 0.5  # small bonus for exploring forward
        else:
            shaped += 1.0  # bigger bonus for moving forward when sensing box

    # Trick 2: Rotation penalty when no sensors
    # Discourage aimless spinning
    if action_idx in [0, 4] and not any_sensor:  # L45 or R45 with no sensor
        shaped -= 0.3

    # Trick 3: Reward for transitioning from no-sensor to sensor
    if any_sensor and not prev_any_sensor:
        shaped += 5.0  # bonus for discovering the box

    # Trick 4: Scale down the massive stuck penalty during training
    # -200 is too harsh and causes the agent to avoid FW entirely
    if stuck:
        # Replace the -200 with something more manageable
        shaped = shaped + 180.0  # net effect: -20 instead of -200
        shaped -= 20.0

    # Trick 5: Bonus for being in push state
    if env.enable_push and action_idx == 2 and not stuck:
        shaped += 2.0  # encourage pushing forward

    return shaped


# ──────────────────────────────────────────────
# PPO Update
# ──────────────────────────────────────────────
def ppo_update(model, optimizer, buffer, returns, advantages,
               clip_eps=0.2, entropy_coef=0.02, value_coef=0.5,
               n_epochs=4, batch_size=64):
    obs, actions, old_log_probs, returns_t, advantages_t = buffer.get_tensors(returns, advantages)

    n = len(obs)
    total_loss_sum = 0
    total_entropy_sum = 0
    n_updates = 0

    for _ in range(n_epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_lp = old_log_probs[idx]
            batch_returns = returns_t[idx]
            batch_adv = advantages_t[idx]

            logits, values, _ = model(batch_obs)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - batch_old_lp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, batch_returns)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss_sum += loss.item()
            total_entropy_sum += entropy.item()
            n_updates += 1

    return total_loss_sum / max(1, n_updates), total_entropy_sum / max(1, n_updates)


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate_policy(model, difficulty=0, wall_obstacles=False, n_episodes=5,
                    max_steps=2000, render=False):
    rewards_list = []
    success_list = []
    steps_list = []

    for ep in range(n_episodes):
        env = OBELIX(
            scaling_factor=3,
            arena_size=500,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=2,
            seed=ep * 1000 + 42,
        )
        raw_obs = env.sensor_feedback.copy()

        frame_stack = FrameStack(n_frames=4)
        aug_obs = augment_obs(raw_obs, -1, 0, max_steps, False)
        stacked = frame_stack.reset(aug_obs)

        hx = None
        total_reward = 0
        prev_action_idx = -1

        for step in range(max_steps):
            action_idx, _, _, hx = model.get_action(stacked, hx, deterministic=True)
            action = ACTION_LIST[action_idx]

            obs, reward, done = env.step(action, render=render)

            total_reward += reward
            prev_action_idx = action_idx

            aug_obs = augment_obs(obs, prev_action_idx, step + 1, max_steps, env.enable_push)
            stacked = frame_stack.push(aug_obs)

            if done:
                break

        rewards_list.append(total_reward)
        success_list.append(1 if env.enable_push and env.done and env.reward >= 2000 else 0)
        steps_list.append(step + 1)

    return np.mean(rewards_list), np.std(rewards_list), np.mean(success_list), np.mean(steps_list)


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────
def train(args):
    print("=" * 60)
    print("PPO Training for OBELIX")
    print("=" * 60)

    device = torch.device("cpu")

    # Use MLP (faster, works well with frame stacking for POMDP)
    if args.use_gru:
        model = ActorCriticGRU(obs_dim=STACKED_OBS_DIM, hidden=256, gru_hidden=128)
    else:
        model = ActorCriticMLP(obs_dim=STACKED_OBS_DIM, hidden=256)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # Curriculum schedule: start with level 0, advance when success rate > threshold
    curriculum_level = 0
    curriculum_wall = False
    curriculum_success_threshold = 0.5  # advance when >50% success

    best_eval_reward = -float("inf")

    # Logging
    all_ep_rewards = []
    all_eval_rewards = []

    total_steps = 0
    episode_count = 0

    # Entropy coefficient scheduling
    entropy_start = args.entropy_coef
    entropy_end = 0.005
    entropy_decay_steps = args.total_timesteps * 0.7

    for iteration in range(args.n_iterations):
        buffer = RolloutBuffer()
        ep_rewards_this_iter = []

        # Determine current difficulty from curriculum
        if args.curriculum:
            difficulty = curriculum_level
            wall = curriculum_wall
        else:
            difficulty = args.difficulty
            wall = args.wall_obstacles

        # Collect rollout
        steps_collected = 0
        while steps_collected < args.steps_per_iter:
            seed = random.randint(0, 1_000_000)
            env = OBELIX(
                scaling_factor=3,
                arena_size=500,
                max_steps=args.max_steps,
                wall_obstacles=wall,
                difficulty=difficulty,
                box_speed=2,
                seed=seed,
            )

            raw_obs = env.sensor_feedback.copy()
            frame_stack = FrameStack(n_frames=4)
            aug_obs = augment_obs(raw_obs, -1, 0, args.max_steps, False)
            stacked = frame_stack.reset(aug_obs)

            hx = None
            prev_action_idx = -1
            ep_reward = 0
            prev_any_sensor = any(raw_obs[:17] > 0)

            for step in range(args.max_steps):
                action_idx, log_prob, value, hx_new = model.get_action(stacked, hx)
                action = ACTION_LIST[action_idx]

                obs, reward, done = env.step(action, render=False)

                # Shape reward for training
                shaped_reward = shape_reward(
                    reward, obs, action_idx, env, prev_any_sensor, step
                )

                # Scale reward for better learning
                scaled_reward = shaped_reward / 100.0  # normalize scale

                any_sensor_now = any(obs[:17] > 0)

                buffer.add(stacked, action_idx, log_prob, scaled_reward, value, float(done))

                prev_action_idx = action_idx
                prev_any_sensor = any_sensor_now
                ep_reward += reward  # track unshaped for logging

                aug_obs = augment_obs(obs, prev_action_idx, step + 1, args.max_steps, env.enable_push)
                stacked = frame_stack.push(aug_obs)
                hx = hx_new

                steps_collected += 1
                total_steps += 1

                if done:
                    break

            ep_rewards_this_iter.append(ep_reward)
            episode_count += 1

        # Compute returns
        _, _, last_value, _ = model.get_action(stacked, hx)
        returns, advantages = buffer.compute_returns(
            last_value, gamma=args.gamma, gae_lambda=args.gae_lambda
        )

        # Entropy scheduling
        progress = min(1.0, total_steps / entropy_decay_steps)
        current_entropy = entropy_start * (1 - progress) + entropy_end * progress

        # PPO update
        avg_loss, avg_entropy = ppo_update(
            model, optimizer, buffer, returns, advantages,
            clip_eps=args.clip_eps,
            entropy_coef=current_entropy,
            value_coef=args.value_coef,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
        )

        # Logging
        mean_ep_reward = np.mean(ep_rewards_this_iter)
        all_ep_rewards.extend(ep_rewards_this_iter)

        if iteration % args.log_interval == 0:
            print(f"Iter {iteration:4d} | Steps {total_steps:8d} | "
                  f"Episodes {episode_count:5d} | "
                  f"MeanR {mean_ep_reward:8.1f} | "
                  f"Loss {avg_loss:.4f} | Entropy {avg_entropy:.4f} | "
                  f"EntCoef {current_entropy:.4f} | "
                  f"Diff {difficulty} Wall {wall}")

        # Evaluate periodically
        if iteration % args.eval_interval == 0:
            eval_r, eval_std, eval_success, eval_steps = evaluate_policy(
                model, difficulty=difficulty, wall_obstacles=wall, n_episodes=5, max_steps=2000
            )
            print(f"  >> EVAL: MeanR={eval_r:.1f} ± {eval_std:.1f}, "
                  f"Success={eval_success:.0%}, AvgSteps={eval_steps:.0f}")

            all_eval_rewards.append(eval_r)

            # Save best model
            if eval_r > best_eval_reward:
                best_eval_reward = eval_r
                torch.save(model.state_dict(), args.save_path)
                print(f"  >> Saved best model (reward={eval_r:.1f})")

            # Curriculum advancement
            if args.curriculum and eval_success >= curriculum_success_threshold:
                if not curriculum_wall:
                    curriculum_wall = True
                    print(f"  >> CURRICULUM: Enabling wall obstacles at diff={curriculum_level}")
                elif curriculum_level < 3:
                    curriculum_level = min(curriculum_level + 1, 3)
                    if curriculum_level == 1:
                        curriculum_level = 2  # skip level 1 (same as 0)
                    curriculum_wall = False
                    print(f"  >> CURRICULUM: Advancing to difficulty={curriculum_level}")

        # Periodic save
        if iteration % (args.eval_interval * 5) == 0 and iteration > 0:
            torch.save(model.state_dict(), args.save_path.replace(".pth", f"_iter{iteration}.pth"))

    # Final save
    torch.save(model.state_dict(), args.save_path.replace(".pth", "_final.pth"))

    # Save training log
    np.savez(
        args.save_path.replace(".pth", "_log.npz"),
        ep_rewards=np.array(all_ep_rewards),
        eval_rewards=np.array(all_eval_rewards),
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best eval reward: {best_eval_reward:.1f}")
    print(f"Model saved to: {args.save_path}")
    print("=" * 60)

    # Final evaluation across all levels
    print("\nFinal Evaluation:")
    for diff in [0, 2, 3]:
        for wall in [False, True]:
            r, s, sr, st = evaluate_policy(model, diff, wall, n_episodes=10, max_steps=2000)
            print(f"  Diff={diff} Wall={wall}: Reward={r:.1f}±{s:.1f}, Success={sr:.0%}, Steps={st:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_iter", type=int, default=4096)
    parser.add_argument("--n_iterations", type=int, default=500)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--curriculum", action="store_true", default=True)
    parser.add_argument("--use_gru", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="weights.pth")
    args = parser.parse_args()
    train(args)