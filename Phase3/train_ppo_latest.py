# ppo_parallel_shaped.py
import os
import math
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 1. ENV WRAPPER: reward shaping + shaped features + action mask
# ============================================================

class ShapedEnv(gym.Wrapper):
    """
    Assumes base env returns:
        obs, reward, terminated, truncated, info

    You should adapt _extract_metrics() to your env.
    """

    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self.prev_potential = 0.0

        # Example: append 4 shaping features to observation
        base_obs_space = env.observation_space
        assert len(base_obs_space.shape) == 1, "Expect flat vector obs"

        low = np.concatenate([
            base_obs_space.low,
            np.array([-np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32)
        ])
        high = np.concatenate([
            base_obs_space.high,
            np.array([ np.inf,  np.inf,  np.inf, 1.0], dtype=np.float32)
        ])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _extract_metrics(self, obs, info):
        """
        Replace this with your task-specific signals.
        Example outputs:
            progress_to_goal: larger is better
            wall_risk: larger means closer to collision
            stuck_score: larger means more stuck
            edge_case_flag: 0/1
        """
        # Example placeholders from info
        progress_to_goal = float(info.get("progress_to_goal", 0.0))
        wall_risk = float(info.get("wall_risk", 0.0))
        stuck_score = float(info.get("stuck_score", 0.0))
        edge_case_flag = float(info.get("edge_case_flag", 0.0))
        return progress_to_goal, wall_risk, stuck_score, edge_case_flag

    def _potential(self, progress_to_goal, wall_risk, stuck_score):
        # Tune these weights for your environment
        return (
            2.0 * progress_to_goal
            - 1.0 * wall_risk
            - 0.5 * stuck_score
        )

    def _augment_obs(self, obs, info):
        progress_to_goal, wall_risk, stuck_score, edge_case_flag = self._extract_metrics(obs, info)
        shaped_feats = np.array(
            [progress_to_goal, wall_risk, stuck_score, edge_case_flag],
            dtype=np.float32
        )
        return np.concatenate([obs.astype(np.float32), shaped_feats], axis=0)

    def _action_mask(self, obs, info):
        """
        Returns 1 for allowed/preferred actions, 0 for strongly discouraged actions.
        Keep this soft at policy level later.
        """
        n = self.action_space.n
        mask = np.ones(n, dtype=np.float32)

        danger_left = float(info.get("danger_left", 0.0))
        danger_right = float(info.get("danger_right", 0.0))
        forward_blocked = float(info.get("forward_blocked", 0.0))

        # Example discrete actions:
        # 0=L45, 1=L22, 2=FW, 3=R22, 4=R45
        if forward_blocked > 0.8:
            mask[2] = 0.0
        if danger_left > 0.8:
            mask[0] = 0.0
            mask[1] = 0.0
        if danger_right > 0.8:
            mask[3] = 0.0
            mask[4] = 0.0

        # Never fully zero the mask
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        progress_to_goal, wall_risk, stuck_score, _ = self._extract_metrics(obs, info)
        self.prev_potential = self._potential(progress_to_goal, wall_risk, stuck_score)

        info = dict(info)
        info["action_mask"] = self._action_mask(obs, info)
        return self._augment_obs(obs, info), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        progress_to_goal, wall_risk, stuck_score, _ = self._extract_metrics(obs, info)
        curr_potential = self._potential(progress_to_goal, wall_risk, stuck_score)

        # Potential-based shaping
        shaping = self.gamma * curr_potential - self.prev_potential
        self.prev_potential = curr_potential

        # Optional extra bounded dense terms
        dense_bonus = 0.05 * progress_to_goal - 0.03 * wall_risk - 0.02 * stuck_score

        shaped_reward = float(base_reward + shaping + dense_bonus)

        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["reward_shaping"] = float(shaping + dense_bonus)
        info["action_mask"] = self._action_mask(obs, info)

        if terminated or truncated:
            self.prev_potential = 0.0

        return self._augment_obs(obs, info), shaped_reward, terminated, truncated, info


# ============================================================
# 2. VECTOR ENV CREATION
# ============================================================

def make_env(env_fn, seed, idx, gamma):
    def thunk():
        env = env_fn()
        env = ShapedEnv(env, gamma=gamma)
        env.reset(seed=seed + idx)
        return env
    return thunk


# ============================================================
# 3. POLICY / VALUE NETWORK WITH ACTION-SHAPING BIAS
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action_mask=None, action=None):
        logits, value = self(x)

        # Soft action shaping:
        # discourage masked actions with a large negative logit
        if action_mask is not None:
            penalty = (1.0 - action_mask) * 8.0
            logits = logits - penalty

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value


# ============================================================
# 4. ROLLOUT BUFFER
# ============================================================

class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, device):
        self.obs = torch.zeros((n_steps, n_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
        self.action_masks = None

    def init_masks(self, n_steps, n_envs, act_dim, device):
        self.action_masks = torch.ones((n_steps, n_envs, act_dim), dtype=torch.float32, device=device)


# ============================================================
# 5. GAE
# ============================================================

def compute_gae(rewards, dones, values, next_value, gamma=0.99, gae_lambda=0.95):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values
    return advantages, returns


# ============================================================
# 6. PPO TRAIN
# ============================================================

def train_ppo(
    env_fn,
    total_timesteps=1_000_000,
    num_envs=8,
    num_steps=256,
    gamma=0.99,
    gae_lambda=0.95,
    lr=3e-4,
    update_epochs=4,
    num_minibatches=8,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=42,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    envs = AsyncVectorEnv([make_env(env_fn, seed, i, gamma) for i in range(num_envs)])
    obs, infos = envs.reset(seed=seed)

    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    act_dim = envs.single_action_space.n
    obs_dim = envs.single_observation_space.shape[0]

    agent = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size

    global_step = 0

    for update in range(1, num_updates + 1):
        buffer = RolloutBuffer(num_steps, num_envs, obs_dim, device)
        buffer.init_masks(num_steps, num_envs, act_dim, device)

        for step in range(num_steps):
            global_step += num_envs
            buffer.obs[step] = obs

            action_mask = np.stack(infos["action_mask"]) if isinstance(infos, dict) else np.stack([i["action_mask"] for i in infos])
            action_mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device)
            buffer.action_masks[step] = action_mask_t

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs, action_mask=action_mask_t)

            buffer.actions[step] = action
            buffer.logprobs[step] = logprob
            buffer.values[step] = value

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated).astype(np.float32)

            buffer.rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            buffer.dones[step] = torch.tensor(done, dtype=torch.float32, device=device)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_action_mask = np.stack(infos["action_mask"]) if isinstance(infos, dict) else np.stack([i["action_mask"] for i in infos])
            next_action_mask_t = torch.tensor(next_action_mask, dtype=torch.float32, device=device)
            _, _, _, next_value = agent.get_action_and_value(obs, action_mask=next_action_mask_t)

        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.dones,
            buffer.values,
            next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        b_obs = buffer.obs.reshape((-1, obs_dim))
        b_actions = buffer.actions.reshape(-1)
        b_logprobs = buffer.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = buffer.values.reshape(-1)
        b_action_masks = buffer.action_masks.reshape((-1, act_dim))

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = inds[start:start + minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    action_mask=b_action_masks[mb_inds],
                    action=b_actions[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        if update % 10 == 0:
            print(
                f"update={update}/{num_updates} "
                f"step={global_step} "
                f"pg={pg_loss.item():.4f} "
                f"v={v_loss.item():.4f} "
                f"ent={ent_loss.item():.4f}"
            )

    envs.close()
    return agent

def env_fn():
    # replace this with your real env
    import gymnasium as gym
    return gym.make("CartPole-v1")


if __name__ == "__main__":
    agent = train_ppo(
        env_fn=env_fn,
        total_timesteps=200_000,
        num_envs=8,
        num_steps=128,
        seed=42,
    )

    torch.save(agent.state_dict(), "ppo_shaped_agent.pt")
    print("training done, saved to ppo_shaped_agent.pt")