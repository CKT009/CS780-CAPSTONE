"""
VPG training with parallel episode collection (multiprocessing).
Collects multiple episodes in parallel before updating, significantly faster on CPU.
"""
import os
import random
import sys
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval import evaluate_agent as eval_agent
from logger import ExperimentLogger
from obelix_fast import OBELIXFast

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_GPU = device.type == "cuda"
print("Device:", device)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
OBS_DIM = 18
STACK = 4
STATE_DIM = OBS_DIM * STACK

DIFFICULTY = 3
ARENA_SIZE = 500
SCALING_FACTOR = 5
MAX_STEPS = 1000
WALL_OBSTACLES = True
BOX_SPEED = 2

GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 1e-4
HIDDEN = 128
ENTROPY_COEF = 0.01
VALUE_COEF = 0.1

EPISODES = 3000
EVAL_EVERY = 120 if ON_GPU else 80
EVAL_RUNS = 10 if ON_GPU else 5

# Parallelization: collect from N episodes in parallel before 1 update
PARALLEL_WORKERS = cpu_count() - 1  # Number of parallel episode collectors
EPISODES_PER_BATCH = PARALLEL_WORKERS  # Collect PARALLEL_WORKERS episodes then update

WEIGHTS_BEST = os.path.join(HERE, "weights_phase3_vpg_parallel_diff3.pth")
WEIGHTS_FINAL = os.path.join(HERE, "weights_phase3_vpg_parallel_diff3_final.pth")

HPARAMS = dict(
    algo="VPG_parallel_phase3",
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    lr=LR,
    hidden=HIDDEN,
    stack=STACK,
    entropy_coef=ENTROPY_COEF,
    value_coef=VALUE_COEF,
    episodes=EPISODES,
    parallel_workers=PARALLEL_WORKERS,
    episodes_per_batch=EPISODES_PER_BATCH,
    difficulty=DIFFICULTY,
    seed=SEED,
    device=str(device),
)


def init_stack(obs):
    s = deque(maxlen=STACK)
    for _ in range(STACK):
        s.append(obs.copy())
    return s


def stack_to_state(stack):
    return np.concatenate(stack, axis=0)


def run_episode_worker(args):
    """
    Worker function: runs one episode, returns trajectory.
    Must be picklable (hence separate function).
    """
    seed, model_state_dict = args
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    env = OBELIXFast(
        scaling_factor=SCALING_FACTOR,
        arena_size=ARENA_SIZE,
        max_steps=MAX_STEPS,
        wall_obstacles=WALL_OBSTACLES,
        difficulty=DIFFICULTY,
        box_speed=BOX_SPEED,
        seed=seed,
    )
    
    # Rebuild model locally
    class VPGActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(STATE_DIM, HIDDEN),
                nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.ReLU(),
            )
            self.policy = nn.Linear(HIDDEN, N_ACTIONS)
            self.value = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            h = self.shared(x)
            logits = self.policy(h)
            value = self.value(h).squeeze(-1)
            return logits, value
    
    model = VPGActorCritic()
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Rollout
    obs = env.reset(seed=seed)
    stack = init_stack(obs)
    done = False

    memory = {
        "states": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "dones": [],
        "values": [],
    }

    ep_raw_total = 0.0

    with torch.no_grad():
        while not done:
            state_vec = stack_to_state(stack)
            x = torch.as_tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            logits, value = model(x)
            logits = logits.squeeze(0)
            value = value.squeeze(0)
            
            dist = torch.distributions.Categorical(logits=logits)
            a_t = dist.sample()
            action = int(a_t.item())
            log_prob = float(dist.log_prob(a_t).item())
            
            next_obs, env_reward, done = env.step(ACTIONS[action], render=False)

            # Reward shaping (minimal)
            reward = float(env_reward)
            if float(obs[17]) == 0.0:  # not stuck
                front_signal = float(np.sum(obs[4:12]) + obs[16])
                if front_signal > 0:
                    reward += 0.10

            memory["states"].append(state_vec)
            memory["actions"].append(action)
            memory["log_probs"].append(log_prob)
            memory["rewards"].append(reward)
            memory["dones"].append(done)
            memory["values"].append(float(value.item()))

            ep_raw_total += float(env_reward)
            obs = next_obs
            stack.append(obs.copy())

    return memory, ep_raw_total


class VPGActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.policy = nn.Linear(HIDDEN, N_ACTIONS)
        self.value = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value


class VPGAgent:
    def __init__(self):
        self.model = VPGActorCritic().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def update(self, memory):
        rewards = np.asarray(memory["rewards"], dtype=np.float32)
        values = np.asarray(memory["values"], dtype=np.float32)
        dones = memory["dones"]

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute advantages via GAE
        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + GAMMA * next_value * mask - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages.insert(0, gae)
            next_value = values[t]

        advantages = np.asarray(advantages, dtype=np.float32)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.as_tensor(np.asarray(memory["states"], dtype=np.float32), device=device)
        actions = torch.as_tensor(memory["actions"], dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor(memory["log_probs"], dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        logits, values_pred = self.model(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_prob = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(new_log_prob * advantages_t).mean()
        value_pred_clipped = values_pred.clamp(-1e6, 1e6)
        ret_clipped = returns_t.clamp(-1e6, 1e6)
        value_loss = torch.mean((ret_clipped - value_pred_clipped) ** 2)
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item()), float(policy_loss.item()), float(value_loss.item()), float(entropy.item())


def build_eval_policy(agent):
    state = {"rng_id": None, "stack": None}

    def policy_fn(obs, rng):
        rid = id(rng)
        if state["rng_id"] != rid or state["stack"] is None:
            state["rng_id"] = rid
            state["stack"] = init_stack(obs)
        else:
            state["stack"].append(obs.copy())

        s = stack_to_state(state["stack"])
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = agent.model(x)
            action_idx = int(torch.argmax(logits.squeeze(0)).item())
        return ACTIONS[action_idx]

    return policy_fn


def train():
    agent = VPGAgent()
    logger = ExperimentLogger("phase3_vpg_parallel_diff3", HPARAMS)

    best_eval = -float("inf")
    train_hist = deque(maxlen=100)
    loss_hist = deque(maxlen=200)

    print(f"Training Phase3 VPG Parallel (diff={DIFFICULTY}, workers={PARALLEL_WORKERS})")

    # Use multiprocessing pool
    pool = Pool(processes=PARALLEL_WORKERS)
    
    total_episodes = 0
    pbar = tqdm(total=EPISODES, desc=f"Phase3 VPG Parallel diff={DIFFICULTY}")

    while total_episodes < EPISODES:
        # Collect EPISODES_PER_BATCH episodes in parallel
        seeds = [random.randint(0, 1_000_000) for _ in range(EPISODES_PER_BATCH)]
        worker_args = [(seed, agent.model.state_dict()) for seed in seeds]
        
        results = pool.map(run_episode_worker, worker_args)
        
        # Aggregate results
        ep_raw_totals = []
        combined_memory = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }
        
        for memory, ep_raw in results:
            for key in combined_memory:
                combined_memory[key].extend(memory[key])
            ep_raw_totals.append(ep_raw)
            train_hist.append(ep_raw)
        
        # Single update on combined batch
        loss, pi_loss, v_loss, ent = agent.update(combined_memory)
        loss_hist.append(loss)
        
        total_episodes += EPISODES_PER_BATCH
        pbar.update(EPISODES_PER_BATCH)

        # Evaluate periodically
        if total_episodes % EVAL_EVERY == 0 or total_episodes == EPISODES_PER_BATCH:
            mean_r, std_r = eval_agent(
                build_eval_policy(agent),
                difficulty=DIFFICULTY,
                runs=EVAL_RUNS,
            )

            print(
                f"\nep={total_episodes:5d}  train_raw={float(np.mean(train_hist)):9.1f}±{float(np.std(train_hist)):6.1f}  "
                f"eval={mean_r:9.1f}+-{std_r:6.1f}  "
                f"loss={float(np.mean(loss_hist)):.4f}"
            )

            logger.log(
                phase=3,
                episode=total_episodes,
                train100=float(np.mean(train_hist)),
                eval_mean=mean_r,
                eval_std=std_r,
                loss=float(np.mean(loss_hist)),
                difficulty=DIFFICULTY,
            )

            if mean_r > best_eval:
                best_eval = mean_r
                torch.save(agent.model.state_dict(), WEIGHTS_BEST)
                print(f"  New best={best_eval:.1f}; saved {WEIGHTS_BEST}")

    pbar.close()
    pool.close()
    pool.join()

    logger.done(best_eval)
    torch.save(agent.model.state_dict(), WEIGHTS_FINAL)
    print(f"Done. Best eval: {best_eval:.1f}")


if __name__ == "__main__":
    train()
