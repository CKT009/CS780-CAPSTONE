import os
import random
import sys
from collections import deque
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
from agent_vpg import policy as vpg_policy

# ========================
# SEED & DEVICE
# ========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ========================
# ENV & HYPERPARAMS
# ========================
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
OBS_DIM = 18
STACK = 8
STATE_DIM = OBS_DIM * STACK
MAX_STEPS = 1000
EPISODES = 2000

GAMMA = 0.99  # CRITICAL: was 0.99999999 (catastrophic for VPG)
GAE_LAMBDA = 0.95
LR = 1e-4
HIDDEN = 128
ENTROPY_COEF = 0.01
VALUE_COEF = 0.1

# Imitation learning (decays over training)
IMITATION_LEARNING = False  # DISABLED: causes gradient conflict early
EXPERT_PROB_START = 0.5
EXPERT_PROB_END = 0.05
BEHAV_CLONE_WEIGHT = 0.2
TOTAL_STEPS = EPISODES * MAX_STEPS

# ========================
# UTILITIES
# ========================
def init_stack(obs):
    s = deque(maxlen=STACK)
    for _ in range(STACK):
        s.append(obs.copy())
    return s

def stack_to_state(stack):
    return np.concatenate(stack, axis=0)

# ========================
# EXPERT POLICY (TEACHER - TRAINING ONLY)
# ========================
def expert_policy(obs):
    """
    Expert policy with improved stuck handling and precautionary logic.
    Sensors: 0-3=left, 4-11=center, 12-15=right, 16=IR, 17=stuck
    """
    stuck = obs[17]
    
    # ===== STUCK HANDLER (PRIORITY 1) =====
    # When stuck, turn toward CLEARER side (lower sensor signal)
    if stuck > 0.5:
        left_signal = np.sum(obs[0:4])
        right_signal = np.sum(obs[12:16])
        
        # Choose clearer side (lower signal = clearer = escape better)
        # With threshold to avoid tiny noise differences
        threshold = 0.2
        if left_signal < right_signal - threshold:
            return 0  # L45 (left is clearer)
        elif right_signal < left_signal - threshold:
            return 4  # R45 (right is clearer)
        else:
            # Sides equally blocked -> aggressive 45-deg turn
            return random.choice([0, 4])  # L45 or R45
    
    # ===== BOX ATTACHED (PRIORITY 2) =====
    if obs[16] > 0.5:
        return 2  # FW (push forward)
    
    # ===== WALL APPROACH WARNING (PRIORITY 3) =====
    # Forward sensors high but not stuck yet = approaching wall
    forward_signal = np.sum(obs[4:12])
    if forward_signal > 2.0:  # Strong forward signal = wall near
        # Warn: rotate gently to find clear path
        left = np.sum(obs[0:4])
        right = np.sum(obs[12:16])
        if left < right - 0.3:
            return 1  # L22 (gentle left)
        elif right < left - 0.3:
            return 3  # R22 (gentle right)
        else:
            return random.choice([1, 3])  # Gentle rotation to find gap
    
    # ===== FORWARD CLEAR (PRIORITY 4) =====
    if forward_signal > 0:
        return 2  # FW
    
    # ===== SIDE CENTERING (PRIORITY 5) =====
    # With relaxed threshold 0.3 instead of 0.5
    left = np.sum(obs[0:4])
    right = np.sum(obs[12:16])
    if left > right + 0.3:
        return 1  # L22 (center to left)
    elif right > left + 0.3:
        return 3  # R22 (center to right)
    
    # ===== FALLBACK (PRIORITY 6) =====
    # If all else fails (forward blocked, sides balanced)
    # Make a small rotation to explore
    return random.choice([1, 3])  # Gentle exploration

# ========================
# SHAPED REWARD
# ========================
def shaped_reward(env_reward, obs, episode_step):
    """SCALED DOWN to [-10, +10] range for VPG stability."""
    r = float(env_reward)
    stuck = obs[17]

def shaped_reward(env_reward, obs, action_idx, rotate_streak, episode_step):
    """SCALED DOWN to [-10, +10] range for VPG stability."""
    r = float(env_reward)
    stuck = obs[17]

    # CRITICAL FIX: reduced scale
    if stuck < 0.5:
        r += 1.0  # was +15
    else:
        penalty = 0.5 if episode_step < MAX_STEPS // 2 else 2.0  # was 20/-60
        r -= penalty

    if obs[16] > 0.5 and stuck < 0.5:  # Box bonus only if NOT stuck
        r += 1.0   # was +100
        r += 0.8   # was +8

    forward_clear = np.sum(obs[4:12])
    if forward_clear > 0.5:
        r += 0.5  # was +5

    left = np.sum(obs[0:4])
    right = np.sum(obs[12:16])
    if abs(left - right) < 1.0:
        r += 0.3  # was +2

    # Rotation penalty (prevent excessive spinning)
    if action_idx in (0, 1, 3, 4) and rotate_streak >= 4:  # ROTATE_PENALTY_START
        r -= 0.35 * (rotate_streak - 4)  # ROTATE_PENALTY_SCALE

    # Small time penalty to encourage efficiency
    r -= 0.005

    return np.clip(r, -10, 10)  # was [-100, 200]

# ========================
# NETWORK
# ========================
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
        return self.policy(h), self.value(h).squeeze(-1)

# ========================
# AGENT
# ========================
class VPGAgent:
    def __init__(self):
        self.model = VPGActorCritic().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.total_steps = 0

    def act(self, state):
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, v = self.model(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return int(action.item()), float(log_prob), float(v)

    def update(self, memory, expert_actions=None):
        rewards = np.array(memory["r"], dtype=np.float32)
        values = np.array(memory["v"], dtype=np.float32)
        dones = np.array(memory["d"], dtype=np.float32)

        # === Reward normalization (PER EPISODE) ===
        # CRITICAL: re-enabled for VPG stability
        if len(rewards) > 1:
            r_mean, r_std = rewards.mean(), rewards.std()
            if r_std > 1e-8:
                rewards = (rewards - r_mean) / r_std

        # === GAE Advantages (corrected) ===
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            # Bootstrap: if episode ended, next_value = 0; else use next step's value
            if t == len(rewards) - 1:
                next_value = 0.0  # Episode ended
            else:
                next_value = values[t + 1]
            
            mask = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * next_value * mask - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === Tensors ===
        states = torch.tensor(np.array(memory["s"]), dtype=torch.float32, device=device)
        actions = torch.tensor(memory["a"], dtype=torch.long, device=device)
        old_logp = torch.tensor(memory["logp"], dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # Forward
        logits, values_pred = self.model(states)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Losses
        policy_loss = -(logp * advantages_t).mean()
        value_loss = ((returns_t - values_pred) ** 2).mean()

        # Total loss (behavioral cloning disabled for pure RL debug)
        total_loss = (
            policy_loss
            + VALUE_COEF * value_loss
            - ENTROPY_COEF * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.total_steps += len(memory["a"])

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def save(self, path):
        torch.save(self.model.state_dict(), path)

# ========================
# TRAINING
# ========================
def train():
    env = OBELIXFast(
        scaling_factor=5,
        arena_size=500,
        max_steps=MAX_STEPS,
        wall_obstacles=True,
        difficulty=3,
        box_speed=2,
    )

    agent = VPGAgent()
    logger = ExperimentLogger(
        algo_name="vpg_diff3",
        hyperparams={
            "lr": LR,
            "hidden": HIDDEN,
            "gamma": GAMMA,
            "entropy_coef": ENTROPY_COEF,
            "value_coef": VALUE_COEF,
            "imitation": IMITATION_LEARNING,
        },
        log_dir="results",
    )

    eval_interval = 50
    eval_episodes = 10
    best_eval = -float("inf")

    print("Starting Phase 3 VPG Training (difficulty=3)")

    for episode in tqdm(range(EPISODES)):
        obs = env.reset(seed=random.randint(0, 1_000_000))
        stack = init_stack(obs)

        memory = {k: [] for k in ["s", "a", "logp", "r", "d", "v", "expert"]}

        done = False
        step = 0

        while not done:
            state = stack_to_state(stack)
            action, log_prob, value = agent.act(state)

            expert_action = expert_policy(obs)

            next_obs, env_reward, done = env.step(ACTIONS[action], render=False)
            reward = shaped_reward(env_reward, obs, step)

            # Store
            memory["s"].append(state)
            memory["a"].append(action)
            memory["logp"].append(log_prob)
            memory["r"].append(reward)
            memory["d"].append(done)
            memory["v"].append(value)
            memory["expert"].append(expert_action)

            obs = next_obs
            stack.append(obs.copy())
            step += 1

        # Update
        losses = agent.update(memory, expert_actions=memory["expert"])

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            eval_rewards = []
            for seed in range(eval_episodes):
                mean_r, std_r = eval_agent(
                    vpg_policy, difficulty=3, runs=1, base_seed=seed
                )
                eval_rewards.append(mean_r)
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)

            # In the periodic evaluation section
            if eval_mean > best_eval:
                best_eval = eval_mean
                agent.save(os.path.join(HERE, "weights_phase3_vpg_diff3.pth"))
                print(f"New best eval: {best_eval:.2f}")

            logger.log(
                phase=3,
                episode=episode + 1,
                train100=np.sum(memory["r"]),  # FIXED: was mean
                eval_mean=eval_mean,
                eval_std=eval_std,
                **losses,
            )

            train_return = np.sum(memory['r'])  # FIXED: was mean
            print(
                f"ep={episode+1:4d} "
                f"train_return={train_return:.1f} "
                f"eval={eval_mean:.1f}±{eval_std:.1f} "
                f"loss={losses['total_loss']:.4f} "
                f"v_loss={losses['value_loss']:.4f} "
                f"ent={losses['entropy']:.4f}"
            )

    # Final save
    final_path = os.path.join(HERE, "weights_phase3_vpg_diff3.pth")
    agent.save(final_path)
    print(f"Training finished. Final weights: {final_path}")

if __name__ == "__main__":
    train()