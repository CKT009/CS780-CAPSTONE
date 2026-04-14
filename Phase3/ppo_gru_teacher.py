import os
import csv
import math
import random
import multiprocessing as mp
from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from obelix_fast import OBELIXFast


# =========================================================
# Config
# =========================================================
@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment
    arena_size: int = 500
    scaling_factor: int = 5
    max_steps: int = 1000
    box_speed: int = 2

    # Parallel rollout
    num_envs: int = 4
    rollout_steps: int = 64

    # Curriculum updates
    stage1_updates: int = 1500   # difficulty 0, no wall
    stage2_updates: int = 1200   # difficulty 2, no wall
    stage3_updates: int = 1800   # difficulty 3, wall

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 1e-4
    clip_coef: float = 0.2
    update_epochs: int = 4
    num_minibatches: int = 4
    entropy_coef: float = 0.03
    value_coef: float = 0.5
    imitation_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True

    # Model
    obs_dim: int = 18
    action_dim: int = 5
    prev_action_dim: int = 5
    prev_reward_dim: int = 1
    hidden_dim: int = 128
    gru_dim: int = 128

    # Reward clipping
    reward_clip: float = 20.0

    # Strict cliff teacher
    use_teacher: bool = True
    teacher_override_prob: float = 1.0
    same_turn_loop_threshold: int = 50

    # Logging / saving
    print_every: int = 10
    save_every: int = 200
    save_dir: str = "checkpoints_ppo_phase3_curriculum"
    run_name: str = "ppo_mem_teacher_curriculum"

    # Eval
    eval_every: int = 100
    eval_runs: int = 3

    # Stage 1 only: tiny exploration assist
    use_stage1_forward_bias: bool = True
    stage1_forward_bias_prob: float = 0.10
    stage1_no_signal_window: int = 10


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot_action(action_idx: int, action_dim: int = 5) -> np.ndarray:
    v = np.zeros(action_dim, dtype=np.float32)
    v[action_idx] = 1.0
    return v


def clip_reward(r: float, limit: float) -> float:
    return float(np.clip(r, -limit, limit))


def build_input(obs: np.ndarray, prev_action_oh: np.ndarray, prev_reward: float) -> np.ndarray:
    pr = np.array([prev_reward], dtype=np.float32)
    return np.concatenate(
        [obs.astype(np.float32), prev_action_oh.astype(np.float32), pr],
        axis=0,
    )


def build_batch_input(
    obs_batch: np.ndarray,
    prev_actions_batch: np.ndarray,
    prev_rewards_batch: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            obs_batch.astype(np.float32),
            prev_actions_batch.astype(np.float32),
            prev_rewards_batch.astype(np.float32),
        ],
        axis=1,
    )


# =========================================================
# Recurrent Actor-Critic
# =========================================================
class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        prev_action_dim: int,
        prev_reward_dim: int,
        hidden_dim: int,
        gru_dim: int,
        action_dim: int,
    ):
        super().__init__()
        input_dim = obs_dim + prev_action_dim + prev_reward_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(hidden_dim, gru_dim, batch_first=True)

        self.actor = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        z = self.encoder(x)
        out, h_new = self.gru(z, h)
        logits = self.actor(out)
        values = self.critic(out)
        return logits, values, h_new

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(1, batch_size, self.gru.hidden_size, device=device)


# =========================================================
# Strict Cliff Teacher
# =========================================================
class CliffTeacher:
    """
    Rare, strong teacher:
    1. stuck => strong recovery
    2. exact same-turn loop => break it
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.recent_actions = deque(maxlen=max(cfg.same_turn_loop_threshold, 16))
        self.recent_obs = deque(maxlen=max(cfg.same_turn_loop_threshold, 16))
        self.escape_plan = deque(maxlen=6)

    def reset(self):
        self.recent_actions.clear()
        self.recent_obs.clear()
        self.escape_plan.clear()

    def update_history(self, obs: np.ndarray, action_idx: int):
        self.recent_actions.append(int(action_idx))
        self.recent_obs.append(obs.copy())

    def _recent_actions_tail(self, k: int):
        if len(self.recent_actions) < k:
            return None
        arr = list(self.recent_actions)
        return arr[-k:]

    def _same_turn_loop(self) -> bool:
        tail = self._recent_actions_tail(self.cfg.same_turn_loop_threshold)
        if tail is None:
            return False
        return (
            all(a == ACTION_TO_IDX["R22"] for a in tail)
            or all(a == ACTION_TO_IDX["L22"] for a in tail)
            or all(a == ACTION_TO_IDX["R45"] for a in tail)
            or all(a == ACTION_TO_IDX["L45"] for a in tail)
        )

    def should_intervene(self, obs: np.ndarray):
        stuck_flag = int(obs[17])
        ir_on = int(obs[16])
        front_bits = obs[4:12]
        front_active = bool(np.any(front_bits > 0.5))

        if len(self.escape_plan) > 0:
            return True, self.escape_plan.popleft(), "escape_plan"

        if stuck_flag == 1:
            if random.random() < 0.5:
                plan = [ACTION_TO_IDX["L45"], ACTION_TO_IDX["L45"], ACTION_TO_IDX["FW"]]
            else:
                plan = [ACTION_TO_IDX["R45"], ACTION_TO_IDX["R45"], ACTION_TO_IDX["FW"]]
            self.escape_plan.extend(plan[1:])
            return True, plan[0], "stuck_recovery_90deg"

        if self._same_turn_loop():
            if front_active or ir_on:
                return True, ACTION_TO_IDX["FW"], "same_turn_commit"
            return True, ACTION_TO_IDX["L45"] if random.random() < 0.5 else ACTION_TO_IDX["R45"], "same_turn_break"

        return False, None, ""


# =========================================================
# Worker process
# =========================================================
def env_worker(remote, parent_remote, env_kwargs: dict):
    parent_remote.close()
    env = OBELIXFast(**env_kwargs)

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == "reset":
                seed = data
                obs = env.reset(seed=seed)
                remote.send(obs)

            elif cmd == "step":
                action = data
                obs, reward, done = env.step(action, render=False)

                if done:
                    obs_reset = env.reset()
                    remote.send((obs, reward, done, obs_reset))
                else:
                    remote.send((obs, reward, done, None))

            elif cmd == "close":
                remote.close()
                break

            else:
                raise ValueError(f"Unknown worker command: {cmd}")

    except KeyboardInterrupt:
        pass


# =========================================================
# Parallel env wrapper
# =========================================================
class ParallelOBELIX:
    def __init__(self, num_envs: int, env_kwargs: dict):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []

        for i in range(num_envs):
            p = mp.Process(
                target=env_worker,
                args=(self.work_remotes[i], self.remotes[i], env_kwargs),
            )
            p.daemon = True
            p.start()
            self.work_remotes[i].close()
            self.processes.append(p)

    def reset(self, seeds=None):
        if seeds is None:
            seeds = [None] * self.num_envs
        for remote, seed in zip(self.remotes, seeds):
            remote.send(("reset", seed))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs, axis=0)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, reset_obs = zip(*results)
        return (
            np.stack(obs, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            list(reset_obs),
        )

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    cfg: Config,
    difficulty: int,
    wall_obstacles: bool,
    runs: int = 3,
):
    device = torch.device(cfg.device)
    model.eval()
    scores = []

    for run in range(runs):
        env = OBELIXFast(
            scaling_factor=cfg.scaling_factor,
            arena_size=cfg.arena_size,
            max_steps=cfg.max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=cfg.box_speed,
            seed=cfg.seed + 10000 + run,
        )

        obs = env.reset(seed=cfg.seed + 10000 + run)
        h = model.init_hidden(1, device)

        prev_action = np.zeros(cfg.action_dim, dtype=np.float32)
        prev_reward = 0.0
        done = False
        total = 0.0

        while not done:
            x_np = build_input(obs, prev_action, prev_reward)
            x = torch.tensor(x_np, dtype=torch.float32, device=device).view(1, 1, -1)

            logits, values, h = model(x, h)
            logits = logits[:, -1, :]
            action_idx = int(torch.argmax(logits, dim=-1).item())
            action = IDX_TO_ACTION[action_idx]

            obs, reward, done = env.step(action, render=False)
            total += float(reward)

            prev_action = one_hot_action(action_idx, cfg.action_dim)
            prev_reward = clip_reward(reward, cfg.reward_clip)

            if done:
                h.zero_()

        scores.append(total)

    model.train()
    return float(np.mean(scores)), float(np.std(scores))


# =========================================================
# GAE
# =========================================================
def compute_gae(
    rewards_t: torch.Tensor,
    dones_t: torch.Tensor,
    values_t: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    """
    rewards_t: [T, N]
    dones_t: [T, N]
    values_t: [T, N]
    next_values: [N]
    """
    T, N = rewards_t.shape
    advantages = torch.zeros_like(rewards_t)
    lastgaelam = torch.zeros(N, dtype=rewards_t.dtype, device=rewards_t.device)

    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - dones_t[t]
            nextvalue = next_values
        else:
            nextnonterminal = 1.0 - dones_t[t]
            nextvalue = values_t[t + 1]

        delta = rewards_t[t] + gamma * nextvalue * nextnonterminal - values_t[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values_t
    return advantages, returns


# =========================================================
# Stage runner
# =========================================================
def run_stage(
    model: nn.Module,
    optimizer: optim.Optimizer,
    cfg: Config,
    stage_name: str,
    difficulty: int,
    wall_obstacles: bool,
    total_updates: int,
    csv_path: str,
):
    device = torch.device(cfg.device)

    env_kwargs = dict(
        scaling_factor=cfg.scaling_factor,
        arena_size=cfg.arena_size,
        max_steps=cfg.max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=cfg.box_speed,
        seed=cfg.seed,
    )

    envs = ParallelOBELIX(cfg.num_envs, env_kwargs)
    teachers = [CliffTeacher(cfg) for _ in range(cfg.num_envs)]

    ep_rewards_live = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_lengths_live = np.zeros(cfg.num_envs, dtype=np.int32)
    completed_rewards = []
    completed_lengths = []

    teacher_intervention_count = 0
    teacher_total_decisions = 0
    stage1_no_signal_counts = np.zeros(cfg.num_envs, dtype=np.int32)

    init_seeds = [cfg.seed + i for i in range(cfg.num_envs)]
    obs = envs.reset(seeds=init_seeds)

    prev_actions = np.zeros((cfg.num_envs, cfg.action_dim), dtype=np.float32)
    prev_rewards = np.zeros((cfg.num_envs, 1), dtype=np.float32)
    h = model.init_hidden(cfg.num_envs, device)

    best_eval_mean = -float("inf")
    best_recent_mean = -float("inf")

    batch_size = cfg.num_envs * cfg.rollout_steps
    minibatch_size = batch_size // cfg.num_minibatches

    pbar = tqdm(range(1, total_updates + 1), desc=stage_name, ncols=170)

    for update in pbar:
        rollout_obs = []
        rollout_prev_actions = []
        rollout_prev_rewards = []
        rollout_actions = []
        rollout_logprobs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []
        rollout_entropies = []

        rollout_logits = []
        rollout_teacher_targets = []
        rollout_teacher_masks = []

        for _ in range(cfg.rollout_steps):
            for i in range(cfg.num_envs):
                prev_action_idx = (
                    int(np.argmax(prev_actions[i]))
                    if prev_actions[i].sum() > 0
                    else ACTION_TO_IDX["FW"]
                )
                teachers[i].update_history(obs[i], prev_action_idx)

            x_np = build_batch_input(obs, prev_actions, prev_rewards)
            x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(1)

            logits, values, h_new = model(x, h)
            logits = logits[:, -1, :]   # [N, A]
            values = values[:, -1, 0]   # [N]

            dist = Categorical(logits=logits)
            sampled_actions_t = dist.sample()

            chosen_actions_idx = sampled_actions_t.detach().cpu().numpy().copy()
            teacher_targets_step = chosen_actions_idx.copy().astype(np.int64)
            teacher_masks_step = np.zeros(cfg.num_envs, dtype=np.float32)

            for i in range(cfg.num_envs):
                teacher_total_decisions += 1
                intervene = False
                teacher_action_idx = None

                if cfg.use_teacher:
                    intervene, teacher_action_idx, reason = teachers[i].should_intervene(obs[i])

                if (
                    stage_name == "stage1"
                    and cfg.use_stage1_forward_bias
                    and not intervene
                ):
                    if np.sum(obs[i][:17]) == 0:
                        stage1_no_signal_counts[i] += 1
                    else:
                        stage1_no_signal_counts[i] = 0

                    if (
                        stage1_no_signal_counts[i] >= cfg.stage1_no_signal_window
                        and random.random() < cfg.stage1_forward_bias_prob
                    ):
                        intervene = True
                        teacher_action_idx = ACTION_TO_IDX["FW"]

                if intervene:
                    teacher_targets_step[i] = int(teacher_action_idx)
                    teacher_masks_step[i] = 1.0
                    if random.random() < cfg.teacher_override_prob:
                        chosen_actions_idx[i] = int(teacher_action_idx)
                        teacher_intervention_count += 1

            chosen_actions_t = torch.tensor(chosen_actions_idx, dtype=torch.long, device=device)
            log_probs_t = dist.log_prob(chosen_actions_t)
            entropies_t = dist.entropy()

            actions_str = [IDX_TO_ACTION[int(a)] for a in chosen_actions_idx]
            next_obs, rewards, dones, reset_obs = envs.step(actions_str)

            clipped_rewards = np.array(
                [clip_reward(r, cfg.reward_clip) for r in rewards],
                dtype=np.float32,
            )

            rollout_obs.append(torch.tensor(obs, dtype=torch.float32, device=device))
            rollout_prev_actions.append(torch.tensor(prev_actions, dtype=torch.float32, device=device))
            rollout_prev_rewards.append(torch.tensor(prev_rewards, dtype=torch.float32, device=device))
            rollout_actions.append(chosen_actions_t)
            rollout_logprobs.append(log_probs_t)
            rollout_values.append(values)
            rollout_rewards.append(torch.tensor(clipped_rewards, dtype=torch.float32, device=device))
            rollout_dones.append(torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=device))
            rollout_entropies.append(entropies_t)

            rollout_logits.append(logits)
            rollout_teacher_targets.append(torch.tensor(teacher_targets_step, dtype=torch.long, device=device))
            rollout_teacher_masks.append(torch.tensor(teacher_masks_step, dtype=torch.float32, device=device))

            ep_rewards_live += rewards
            ep_lengths_live += 1

            new_prev_actions = np.zeros_like(prev_actions, dtype=np.float32)
            for i, a in enumerate(chosen_actions_idx):
                new_prev_actions[i, int(a)] = 1.0
            new_prev_rewards = clipped_rewards.reshape(-1, 1)

            for i in range(cfg.num_envs):
                if dones[i]:
                    completed_rewards.append(float(ep_rewards_live[i]))
                    completed_lengths.append(int(ep_lengths_live[i]))
                    ep_rewards_live[i] = 0.0
                    ep_lengths_live[i] = 0
                    stage1_no_signal_counts[i] = 0

                    teachers[i].reset()
                    h_new[:, i, :] = 0.0
                    new_prev_actions[i] = 0.0
                    new_prev_rewards[i] = 0.0

                    if reset_obs[i] is not None:
                        next_obs[i] = reset_obs[i]

            obs = next_obs
            prev_actions = new_prev_actions
            prev_rewards = new_prev_rewards
            h = h_new.detach()

        x_np = build_batch_input(obs, prev_actions, prev_rewards)
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            _, next_values, _ = model(x, h)
            next_values = next_values[:, -1, 0]

        obs_t = torch.stack(rollout_obs).detach()                    # [T, N, obs]
        prev_actions_t = torch.stack(rollout_prev_actions).detach() # [T, N, 5]
        prev_rewards_t = torch.stack(rollout_prev_rewards).detach() # [T, N, 1]
        actions_t = torch.stack(rollout_actions).detach()           # [T, N]
        old_logprobs_t = torch.stack(rollout_logprobs).detach()     # [T, N]
        values_t = torch.stack(rollout_values).detach()             # [T, N]
        rewards_t = torch.stack(rollout_rewards).detach()           # [T, N]
        dones_t = torch.stack(rollout_dones).detach()               # [T, N]

        teacher_targets_t = torch.stack(rollout_teacher_targets).detach()  # [T, N]
        teacher_masks_t = torch.stack(rollout_teacher_masks).detach()      # [T, N]

        advantages_t, returns_t = compute_gae(
            rewards_t=rewards_t,
            dones_t=dones_t,
            values_t=values_t,
            next_values=next_values,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        
        advantages_t = advantages_t.detach()
        returns_t = returns_t.detach()
        next_values = next_values.detach()

        if cfg.normalize_advantages:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        b_obs = obs_t.reshape(-1, cfg.obs_dim)
        b_prev_actions = prev_actions_t.reshape(-1, cfg.action_dim)
        b_prev_rewards = prev_rewards_t.reshape(-1, 1)
        b_actions = actions_t.reshape(-1)
        b_old_logprobs = old_logprobs_t.reshape(-1)
        b_advantages = advantages_t.reshape(-1)
        b_returns = returns_t.reshape(-1)
        b_old_values = values_t.reshape(-1)

        b_teacher_targets = teacher_targets_t.reshape(-1)
        b_teacher_masks = teacher_masks_t.reshape(-1)

        inds = np.arange(batch_size)

        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_entropy = []
        epoch_imitation_losses = []
        epoch_total_losses = []

        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)

            for start in range(0, batch_size, minibatch_size):
                mb_inds = inds[start:start + minibatch_size]

                mb_x_np = np.concatenate(
                    [
                        b_obs[mb_inds].detach().cpu().numpy(),
                        b_prev_actions[mb_inds].detach().cpu().numpy(),
                        b_prev_rewards[mb_inds].detach().cpu().numpy(),
                    ],
                    axis=1,
                )
                mb_x = torch.tensor(mb_x_np, dtype=torch.float32, device=device).unsqueeze(1)
                mb_h = model.init_hidden(len(mb_inds), device)

                new_logits, new_values, _ = model(mb_x, mb_h)
                new_logits = new_logits[:, -1, :]
                new_values = new_values[:, -1, 0]

                new_dist = Categorical(logits=new_logits)
                new_logprobs = new_dist.log_prob(b_actions[mb_inds])
                entropy = new_dist.entropy().mean()

                logratio = new_logprobs - b_old_logprobs[mb_inds]
                ratio = logratio.exp()

                pg_loss1 = -b_advantages[mb_inds] * ratio
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(
                    ratio,
                    1.0 - cfg.clip_coef,
                    1.0 + cfg.clip_coef,
                )
                actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_pred = new_values
                value_pred_clipped = b_old_values[mb_inds] + torch.clamp(
                    value_pred - b_old_values[mb_inds],
                    -cfg.clip_coef,
                    cfg.clip_coef,
                )
                value_losses = (value_pred - b_returns[mb_inds]) ** 2
                value_losses_clipped = (value_pred_clipped - b_returns[mb_inds]) ** 2
                critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                ce_all = F.cross_entropy(
                    new_logits,
                    b_teacher_targets[mb_inds],
                    reduction="none",
                )
                mask = b_teacher_masks[mb_inds]
                imitation_loss = (ce_all * mask).sum() / mask.sum().clamp(min=1.0)

                loss = (
                    actor_loss
                    + cfg.value_coef * critic_loss
                    - cfg.entropy_coef * entropy
                    + cfg.imitation_coef * imitation_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                epoch_actor_losses.append(float(actor_loss.item()))
                epoch_critic_losses.append(float(critic_loss.item()))
                epoch_entropy.append(float(entropy.item()))
                epoch_imitation_losses.append(float(imitation_loss.item()))
                epoch_total_losses.append(float(loss.item()))

        recent_rewards = completed_rewards[-50:] if completed_rewards else [0.0]
        recent_lengths = completed_lengths[-50:] if completed_lengths else [0]
        mean_reward_recent = float(np.mean(recent_rewards))
        mean_length_recent = float(np.mean(recent_lengths))
        intervention_rate = teacher_intervention_count / max(1, teacher_total_decisions)
        teacher_mask_fraction = float(b_teacher_masks.mean().item())

        eval_mean = float("nan")
        eval_std = float("nan")
        if update % cfg.eval_every == 0:
            eval_mean, eval_std = evaluate_model(
                model=model,
                cfg=cfg,
                difficulty=difficulty,
                wall_obstacles=wall_obstacles,
                runs=cfg.eval_runs,
            )
            if eval_mean > best_eval_mean:
                best_eval_mean = eval_mean
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.save_dir, f"best_eval_{cfg.run_name}_{stage_name}.pth"),
                )

        if mean_reward_recent > best_recent_mean:
            best_recent_mean = mean_reward_recent
            torch.save(
                model.state_dict(),
                os.path.join(cfg.save_dir, f"best_recent_{cfg.run_name}_{stage_name}.pth"),
            )

        if update % cfg.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(cfg.save_dir, f"{cfg.run_name}_{stage_name}_update_{update}.pth"),
            )

        actor_loss_mean = float(np.mean(epoch_actor_losses))
        critic_loss_mean = float(np.mean(epoch_critic_losses))
        entropy_mean = float(np.mean(epoch_entropy))
        imitation_loss_mean = float(np.mean(epoch_imitation_losses))
        total_loss_mean = float(np.mean(epoch_total_losses))

        if update % cfg.print_every == 0:
            pbar.set_postfix({
                "EpDone": len(completed_rewards),
                "MeanR": f"{mean_reward_recent:.0f}",
                "Len": f"{mean_length_recent:.0f}",
                "Loss": f"{total_loss_mean:.1e}",
                "A": f"{actor_loss_mean:.2f}",
                "C": f"{critic_loss_mean:.2f}",
                "Ent": f"{entropy_mean:.2f}",
                "Imit": f"{imitation_loss_mean:.2f}",
                "Teach": f"{100.0 * intervention_rate:.1f}%",
                "Mask": f"{100.0 * teacher_mask_fraction:.1f}%",
            })

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                stage_name,
                difficulty,
                wall_obstacles,
                update,
                len(completed_rewards),
                mean_reward_recent,
                mean_length_recent,
                actor_loss_mean,
                critic_loss_mean,
                entropy_mean,
                imitation_loss_mean,
                total_loss_mean,
                float(intervention_rate),
                float(teacher_mask_fraction),
                eval_mean,
                eval_std,
            ])

    envs.close()
    return model


# =========================================================
# Main curriculum
# =========================================================
def train_curriculum(cfg: Config):
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    model = RecurrentActorCritic(
        obs_dim=cfg.obs_dim,
        prev_action_dim=cfg.prev_action_dim,
        prev_reward_dim=cfg.prev_reward_dim,
        hidden_dim=cfg.hidden_dim,
        gru_dim=cfg.gru_dim,
        action_dim=cfg.action_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    csv_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_training_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage",
            "difficulty",
            "wall_obstacles",
            "update",
            "episodes_finished",
            "mean_reward_recent",
            "mean_length_recent",
            "actor_loss",
            "critic_loss",
            "entropy",
            "imitation_loss",
            "total_loss",
            "teacher_intervention_rate",
            "teacher_mask_fraction",
            "eval_mean",
            "eval_std",
        ])

    print("\n=== Stage 1: difficulty=0, wall=False ===")
    model = run_stage(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        stage_name="stage1",
        difficulty=0,
        wall_obstacles=False,
        total_updates=cfg.stage1_updates,
        csv_path=csv_path,
    )
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.run_name}_after_stage1.pth"))

    print("\n=== Stage 2: difficulty=2, wall=False ===")
    model = run_stage(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        stage_name="stage2",
        difficulty=2,
        wall_obstacles=False,
        total_updates=cfg.stage2_updates,
        csv_path=csv_path,
    )
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.run_name}_after_stage2.pth"))

    print("\n=== Stage 3: difficulty=3, wall=True ===")
    model = run_stage(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        stage_name="stage3",
        difficulty=3,
        wall_obstacles=True,
        total_updates=cfg.stage3_updates,
        csv_path=csv_path,
    )
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.run_name}_final.pth"))

    return model


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = Config()
    train_curriculum(cfg)