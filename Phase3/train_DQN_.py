import os
import csv
import math
import random
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from obelix_fast import OBELIXFast


# =========================================================
# Config
# =========================================================
@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Env
    arena_size: int = 500
    scaling_factor: int = 5
    max_steps: int = 1000
    wall_obstacles: bool = True
    difficulty: int = 3
    box_speed: int = 2

    # Observation setup
    obs_dim: int = 18
    stack_len: int = 4
    action_dim: int = 5

    # Teacher data collection
    teacher_episodes: int = 2500

    # Supervised training
    train_split: float = 0.9
    hidden_dim: int = 256
    hidden_dim_2: int = 256
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20

    # DAgger
    dagger_rounds: int = 3
    dagger_episodes_per_round: int = 300
    dagger_mixture_beta_start: float = 0.7
    dagger_mixture_beta_decay: float = 0.5

    # Logging / saving
    save_dir: str = "bc_dagger_phase3"
    dataset_csv: str = "teacher_dataset.csv"
    model_name: str = "bc_dagger_phase3_best.pth"

    # Eval
    eval_runs: int = 20


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


class ObsStacker:
    def __init__(self, obs_dim: int, stack_len: int):
        self.obs_dim = obs_dim
        self.stack_len = stack_len
        self.frames = deque(maxlen=stack_len)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.stack_len):
            self.frames.append(obs.astype(np.float32).copy())
        return self.get()

    def append(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs.astype(np.float32).copy())
        return self.get()

    def get(self) -> np.ndarray:
        assert len(self.frames) == self.stack_len
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


# =========================================================
# Teacher
# =========================================================
class TeacherPolicy:
    """
    Strong heuristic teacher for deadline mode.
    Goal: produce usable trajectories, not theoretical purity.

    Obs conventions used:
    - obs[16] : IR bit
    - obs[17] : stuck bit
    - obs[4:12] : front sonar region
    """

    def __init__(self):
        self.recent_actions = deque(maxlen=12)
        self.recent_obs = deque(maxlen=12)
        self.escape_plan = deque(maxlen=6)
        self.no_signal_count = 0
        self.search_phase = 0

    def reset(self):
        self.recent_actions.clear()
        self.recent_obs.clear()
        self.escape_plan.clear()
        self.no_signal_count = 0
        self.search_phase = 0

    def update(self, obs: np.ndarray, last_action_idx: int):
        self.recent_actions.append(int(last_action_idx))
        self.recent_obs.append(obs.copy())

    def _same_turn_loop(self) -> bool:
        if len(self.recent_actions) < 10:
            return False
        tail = list(self.recent_actions)[-10:]
        return (
            all(a == ACTION_TO_IDX["R22"] for a in tail)
            or all(a == ACTION_TO_IDX["L22"] for a in tail)
            or all(a == ACTION_TO_IDX["R45"] for a in tail)
            or all(a == ACTION_TO_IDX["L45"] for a in tail)
        )

    def _sensor_groups(self, obs: np.ndarray):
        # crude spatial grouping on 16 non-IR/non-stuck bits
        # left, front, right groups
        left = float(np.sum(obs[0:5]))
        front = float(np.sum(obs[5:11]))
        right = float(np.sum(obs[11:16]))
        return left, front, right

    def act(self, obs: np.ndarray) -> int:
        stuck_flag = int(obs[17])
        ir_on = int(obs[16])

        left, front, right = self._sensor_groups(obs)
        total_signal = left + front + right
        front_active = front > 0.5

        # Continue escape macro
        if len(self.escape_plan) > 0:
            return self.escape_plan.popleft()

        # Hard recovery if stuck
        if stuck_flag == 1:
            if random.random() < 0.5:
                plan = [ACTION_TO_IDX["L45"], ACTION_TO_IDX["L45"], ACTION_TO_IDX["FW"]]
            else:
                plan = [ACTION_TO_IDX["R45"], ACTION_TO_IDX["R45"], ACTION_TO_IDX["FW"]]
            self.escape_plan.extend(plan[1:])
            return plan[0]

        # Break exact turn loops hard
        if self._same_turn_loop():
            if front_active or ir_on:
                return ACTION_TO_IDX["FW"]
            return ACTION_TO_IDX["L45"] if random.random() < 0.5 else ACTION_TO_IDX["R45"]

        # Strong front / IR signal -> commit forward
        if ir_on or front >= max(left, right) and front > 1.0:
            return ACTION_TO_IDX["FW"]

        # Turn toward stronger side if asymmetric
        if left > right + 0.75:
            return ACTION_TO_IDX["L22"] if left < 3.0 else ACTION_TO_IDX["L45"]
        if right > left + 0.75:
            return ACTION_TO_IDX["R22"] if right < 3.0 else ACTION_TO_IDX["R45"]

        # Some weak signal ahead -> probe forward
        if total_signal > 0.0 and front > 0.0:
            return ACTION_TO_IDX["FW"]

        # No signal: structured search, not random spin
        if total_signal == 0.0:
            self.no_signal_count += 1
        else:
            self.no_signal_count = 0

        if self.no_signal_count < 4:
            return ACTION_TO_IDX["FW"]

        # search cycle: two small turns, forward burst, swap direction
        phase = self.search_phase % 8
        if phase in [0, 1]:
            act = ACTION_TO_IDX["L22"]
        elif phase in [2, 3]:
            act = ACTION_TO_IDX["FW"]
        elif phase in [4, 5]:
            act = ACTION_TO_IDX["R22"]
        else:
            act = ACTION_TO_IDX["FW"]

        self.search_phase += 1
        return act


# =========================================================
# Dataset
# =========================================================
class BCExampleStore:
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []

    def add(self, state: np.ndarray, action: int):
        self.states.append(state.astype(np.float32).copy())
        self.actions.append(int(action))

    def extend(self, states: List[np.ndarray], actions: List[int]):
        for s, a in zip(states, actions):
            self.add(s, a)

    def to_numpy(self):
        x = np.stack(self.states, axis=0).astype(np.float32)
        y = np.array(self.actions, dtype=np.int64)
        return x, y

    def __len__(self):
        return len(self.actions)


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# =========================================================
# Model
# =========================================================
class BCPolicyNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, hidden_dim_2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================
# Data collection
# =========================================================
def collect_teacher_dataset(cfg: Config) -> BCExampleStore:
    env = OBELIXFast(
        scaling_factor=cfg.scaling_factor,
        arena_size=cfg.arena_size,
        max_steps=cfg.max_steps,
        wall_obstacles=cfg.wall_obstacles,
        difficulty=cfg.difficulty,
        box_speed=cfg.box_speed,
        seed=cfg.seed,
    )

    teacher = TeacherPolicy()
    store = BCExampleStore()

    for ep in tqdm(range(cfg.teacher_episodes), desc="Collect teacher data", ncols=120):
        obs = env.reset(seed=cfg.seed + ep)
        stacker = ObsStacker(cfg.obs_dim, cfg.stack_len)
        state = stacker.reset(obs)

        teacher.reset()
        done = False
        last_action = ACTION_TO_IDX["FW"]

        while not done:
            teacher.update(obs, last_action)
            teacher_action = teacher.act(obs)

            store.add(state, teacher_action)

            action = IDX_TO_ACTION[teacher_action]
            next_obs, reward, done = env.step(action, render=False)

            obs = next_obs
            state = stacker.append(obs)
            last_action = teacher_action

    return store


# =========================================================
# Supervised training
# =========================================================
def train_bc_model(cfg: Config, store: BCExampleStore) -> BCPolicyNet:
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device)

    x, y = store.to_numpy()

    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)

    split = int(cfg.train_split * n)
    train_idx = idx[:split]
    val_idx = idx[split:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    train_ds = ArrayDataset(x_train, y_train)
    val_ds = ArrayDataset(x_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = BCPolicyNet(
        input_dim=cfg.obs_dim * cfg.stack_len,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        hidden_dim_2=cfg.hidden_dim_2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = os.path.join(cfg.save_dir, cfg.model_name)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += int((preds == yb).sum().item())
            train_total += int(xb.size(0))

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += float(loss.item()) * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += int((preds == yb).sum().item())
                val_total += int(xb.size(0))

        train_loss /= max(1, train_total)
        val_loss /= max(1, val_total)
        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch:02d} | "
            f"TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} | "
            f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


# =========================================================
# DAgger data collection
# =========================================================
@torch.no_grad()
def collect_dagger_data(cfg: Config, model: BCPolicyNet, round_idx: int) -> BCExampleStore:
    env = OBELIXFast(
        scaling_factor=cfg.scaling_factor,
        arena_size=cfg.arena_size,
        max_steps=cfg.max_steps,
        wall_obstacles=cfg.wall_obstacles,
        difficulty=cfg.difficulty,
        box_speed=cfg.box_speed,
        seed=cfg.seed + 100000 + round_idx,
    )

    device = torch.device(cfg.device)
    teacher = TeacherPolicy()
    store = BCExampleStore()

    beta = cfg.dagger_mixture_beta_start * (cfg.dagger_mixture_beta_decay ** round_idx)

    for ep in tqdm(range(cfg.dagger_episodes_per_round), desc=f"DAgger round {round_idx+1}", ncols=120):
        obs = env.reset(seed=cfg.seed + 100000 + round_idx * 10000 + ep)
        stacker = ObsStacker(cfg.obs_dim, cfg.stack_len)
        state = stacker.reset(obs)

        teacher.reset()
        done = False
        last_action = ACTION_TO_IDX["FW"]

        while not done:
            teacher.update(obs, last_action)
            teacher_action = teacher.act(obs)

            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(state_t).squeeze(0)
            student_action = int(torch.argmax(logits).item())

            # DAgger label is always teacher action
            store.add(state, teacher_action)

            # Mixture policy for rollout
            if random.random() < beta:
                chosen_action = teacher_action
            else:
                chosen_action = student_action

            action = IDX_TO_ACTION[chosen_action]
            next_obs, reward, done = env.step(action, render=False)

            obs = next_obs
            state = stacker.append(obs)
            last_action = chosen_action

    return store


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_model(cfg: Config, model: BCPolicyNet, runs: int) -> Tuple[float, float, float]:
    device = torch.device(cfg.device)
    env = OBELIXFast(
        scaling_factor=cfg.scaling_factor,
        arena_size=cfg.arena_size,
        max_steps=cfg.max_steps,
        wall_obstacles=cfg.wall_obstacles,
        difficulty=cfg.difficulty,
        box_speed=cfg.box_speed,
        seed=cfg.seed + 99999,
    )

    scores = []
    lengths = []

    model.eval()

    for run in range(runs):
        obs = env.reset(seed=cfg.seed + 99999 + run)
        stacker = ObsStacker(cfg.obs_dim, cfg.stack_len)
        state = stacker.reset(obs)

        done = False
        total_reward = 0.0
        ep_len = 0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(state_t).squeeze(0)
            action_idx = int(torch.argmax(logits).item())

            action = IDX_TO_ACTION[action_idx]
            next_obs, reward, done = env.step(action, render=False)

            total_reward += float(reward)
            ep_len += 1
            obs = next_obs
            state = stacker.append(obs)

        scores.append(total_reward)
        lengths.append(ep_len)

    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(lengths))


# =========================================================
# Save dataset CSV
# =========================================================
def save_dataset_csv(cfg: Config, store: BCExampleStore):
    path = os.path.join(cfg.save_dir, cfg.dataset_csv)
    x, y = store.to_numpy()

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"s{i}" for i in range(x.shape[1])] + ["action_idx", "action_name"]
        writer.writerow(header)
        for row, a in zip(x, y):
            writer.writerow(list(row.astype(float)) + [int(a), IDX_TO_ACTION[int(a)]])


# =========================================================
# Main
# =========================================================
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)

    print("\n=== Collect teacher dataset ===")
    store = collect_teacher_dataset(cfg)
    print(f"Collected {len(store)} examples")

    save_dataset_csv(cfg, store)

    print("\n=== Train initial BC model ===")
    model = train_bc_model(cfg, store)

    mean_r, std_r, mean_len = evaluate_model(cfg, model, cfg.eval_runs)
    print(f"Initial BC eval | MeanR {mean_r:.2f} | StdR {std_r:.2f} | MeanLen {mean_len:.2f}")

    for r in range(cfg.dagger_rounds):
        print(f"\n=== DAgger round {r+1}/{cfg.dagger_rounds} ===")
        new_store = collect_dagger_data(cfg, model, r)
        print(f"Collected {len(new_store)} new labeled examples")
        store.extend(new_store.states, new_store.actions)

        save_dataset_csv(cfg, store)
        model = train_bc_model(cfg, store)

        mean_r, std_r, mean_len = evaluate_model(cfg, model, cfg.eval_runs)
        print(f"After DAgger {r+1} | MeanR {mean_r:.2f} | StdR {std_r:.2f} | MeanLen {mean_len:.2f}")

    final_path = os.path.join(cfg.save_dir, cfg.model_name)
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model to: {final_path}")


if __name__ == "__main__":
    main()