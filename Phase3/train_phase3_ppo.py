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
from obelix import OBELIX
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
STACK = 8
STATE_DIM = OBS_DIM * STACK

DIFFICULTY = 3
ARENA_SIZE = 500
SCALING_FACTOR = 5
MAX_STEPS = 1000
WALL_OBSTACLES = True
BOX_SPEED = 2
USE_FAST_ENV = True

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 2e-4  # Lowered from 3e-4 for CPU stability
HIDDEN = 192

EPOCHS = 8
BATCH_SIZE = 128
ENTROPY_COEF = 0.05  # High entropy for hard exploration task
VALUE_COEF = 0.1  # REDUCED from 0.5 (value loss was dominating)
GRAD_CLIP = 5.0

EPISODES = 2000
EVAL_EVERY = 25
EVAL_RUNS = 3

ROTATE_PENALTY_START = 4
ROTATE_PENALTY_SCALE = 0.35
FORWARD_BONUS = 0.20
FRONT_PROGRESS_BONUS = 0.10

# Imitation learning (expert teacher decays over training)
IMITATION_LEARNING = False  
EXPERT_PROB_START = 0.5
EXPERT_PROB_END = 0.05
BEHAV_CLONE_WEIGHT = 0.0

WEIGHTS_BEST = os.path.join(HERE, "weights_phase3_ppo_diff3.pth")
WEIGHTS_FINAL = os.path.join(HERE, "weights_phase3_ppo_diff3_final.pth")

HPARAMS = dict(
    algo="PPO_phase3_expert",
    gamma=GAMMA,
    gae_lambda=LAMBDA,
    clip_eps=CLIP_EPS,
    lr=LR,
    hidden=HIDDEN,
    stack=STACK,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    entropy_coef=ENTROPY_COEF,
    value_coef=VALUE_COEF,
    grad_clip=GRAD_CLIP,
    episodes=EPISODES,
    eval_every=EVAL_EVERY,
    eval_runs=EVAL_RUNS,
    rotate_penalty_start=ROTATE_PENALTY_START,
    rotate_penalty_scale=ROTATE_PENALTY_SCALE,
    forward_bonus=FORWARD_BONUS,
    front_progress_bonus=FRONT_PROGRESS_BONUS,
    arena_size=ARENA_SIZE,
    scaling_factor=SCALING_FACTOR,
    max_steps=MAX_STEPS,
    wall_obstacles=WALL_OBSTACLES,
    difficulty=DIFFICULTY,
    box_speed=BOX_SPEED,
    use_fast_env=USE_FAST_ENV,
    seed=SEED,
    device=str(device),
    imitation_learning=IMITATION_LEARNING,
    expert_prob_start=EXPERT_PROB_START,
    expert_prob_end=EXPERT_PROB_END,
    behav_clone_weight=BEHAV_CLONE_WEIGHT,
)


def init_stack(obs):
    s = deque(maxlen=STACK)
    for _ in range(STACK):
        s.append(obs.copy())
    return s


def stack_to_state(stack):
    return np.concatenate(stack, axis=0)


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

def shaped_reward(env_reward, obs, action_idx, rotate_streak, prev_front_signal, episode_step):
    """
    DENSE rewards for stable value learning.
    Multiple small signals instead of sparse big signals.
    """
    r = float(env_reward)
    stuck = obs[17]

    # ===== DENSE SIGNALS (smooth value function) =====
    
    # Per-step base reward (mostly neutral)
    r += 0.0  # No step cost, no step bonus
    
    # Forward action bonus (encourages progress toward goal)
    if action_idx == 2:
        r += 0.1
    
    # Forward sensor bonus (you're approaching something)
    forward_signal = np.sum(obs[4:12])
    r += 0.01 * forward_signal  # Tiny bonus for being near stuff
    
    # Box attachment (goal signal - moderate, not extreme)
    if obs[16] > 0.5:
        r += 1.0  # Goal bonus (not 5.0 - too volatile)
    
    # Stuck penalty (very small)
    if stuck > 0.5:
        r -= 0.01

    front_signal = float(np.sum(obs[4:12]) + obs[16])
    
    return np.clip(r, -2, 2), front_signal  # Tighter clipping for stability


class PPOActorCritic(nn.Module):
    def __init__(self, in_dim=STATE_DIM, hidden=HIDDEN, n_actions=N_ACTIONS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(self):
        self.model = PPOActorCritic().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def act(self, state, greedy=False):
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = self.model(x)
            logits = logits.squeeze(0)
            value = value.squeeze(0)
            if greedy:
                action = int(torch.argmax(logits).item())
                log_prob = 0.0
            else:
                dist = torch.distributions.Categorical(logits=logits)
                a_t = dist.sample()
                action = int(a_t.item())
                log_prob = float(dist.log_prob(a_t).item())
        return action, log_prob, float(value.item())

    def update(self, memory, expert_actions=None):
        rewards = np.asarray(memory["rewards"], dtype=np.float32)
        values = np.asarray(memory["values"], dtype=np.float32)
        dones = np.asarray(memory["dones"], dtype=np.float32)

        # === Reward normalization (PER EPISODE) ===
        # CRITICAL: re-enabled for PPO stability
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
            
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + GAMMA * next_value * mask - values[t]
            gae = delta + GAMMA * LAMBDA * mask * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clip extreme returns to prevent value explosion
        returns = np.clip(returns, -100, 100)

        states = torch.as_tensor(np.asarray(memory["states"], dtype=np.float32), device=device)
        actions = torch.as_tensor(memory["actions"], dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor(memory["log_probs"], dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=device)

        losses = []
        policy_losses = []
        value_losses = []
        entropies = []

        n = len(states)
        for _ in range(EPOCHS):
            idx = np.random.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                b = idx[start : start + BATCH_SIZE]

                s = states[b]
                a = actions[b]
                old_lp = old_log_probs[b]
                adv = advantages_t[b]
                ret = returns_t[b]

                logits, values_pred = self.model(s)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(a)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_prob - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv

                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with PROPER PPO-style clipping (prevent divergence)
                # Clip around old values, not predictions
                value_clipped = values_t[b] + (values_pred - values_t[b]).clamp(-CLIP_EPS, CLIP_EPS)
                value_loss1 = (ret - values_pred) ** 2
                value_loss2 = (ret - value_clipped) ** 2
                value_loss = torch.max(value_loss1, value_loss2).mean()

                # Total loss with behavioral cloning (light imitation guidance)
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                
                if IMITATION_LEARNING and expert_actions is not None:
                    expert_actions_t = torch.as_tensor(expert_actions, dtype=torch.long, device=device)
                    expert_a = expert_actions_t[b]
                    bc_loss = torch.nn.functional.cross_entropy(logits, expert_a)
                    loss = loss + BEHAV_CLONE_WEIGHT * bc_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()

                losses.append(float(loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        return (
            float(np.mean(losses)),
            float(np.mean(policy_losses)),
            float(np.mean(value_losses)),
            float(np.mean(entropies)),
        )


def build_eval_policy(agent):
    # Eval API does not provide done/reset callbacks, so we detect episode boundaries by RNG object.
    state = {"rng_id": None, "stack": None}

    def policy_fn(obs, rng):
        rid = id(rng)
        if state["rng_id"] != rid or state["stack"] is None:
            state["rng_id"] = rid
            state["stack"] = init_stack(obs)
        else:
            state["stack"].append(obs.copy())

        s = stack_to_state(state["stack"])
        action, _, _ = agent.act(s, greedy=True)
        return ACTIONS[action]

    return policy_fn


def train():
    env_cls = OBELIXFast if USE_FAST_ENV else OBELIX
    env = env_cls(
        scaling_factor=SCALING_FACTOR,
        arena_size=ARENA_SIZE,
        max_steps=MAX_STEPS,
        wall_obstacles=WALL_OBSTACLES,
        difficulty=DIFFICULTY,
        box_speed=BOX_SPEED,
        seed=SEED,
    )

    agent = PPOAgent()
    logger = ExperimentLogger("phase3_ppo_diff3", HPARAMS)

    best_eval = -float("inf")
    train_hist = deque(maxlen=100)
    shaped_hist = deque(maxlen=100)
    loss_hist = deque(maxlen=200)
    pi_hist = deque(maxlen=200)
    v_hist = deque(maxlen=200)
    ent_hist = deque(maxlen=200)

    print(
        f"Training Phase3 PPO (diff={DIFFICULTY}, env={'OBELIXFast' if USE_FAST_ENV else 'OBELIX'})"
    )
    print(f"  Pure RL (no BC) | Entropy: {ENTROPY_COEF} | Gradient clip: {GRAD_CLIP}")

    for ep in tqdm(range(EPISODES), desc="Phase3 PPO diff=3"):
        obs = env.reset(seed=random.randint(0, 1_000_000))
        stack = init_stack(obs)
        done = False

        memory = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "expert": [],  # For imitation learning
        }
        ep_step = 0

        ep_raw_total = 0.0
        ep_shaped_total = 0.0
        ep_actions = []  # DEBUG: track action distribution
        rotate_streak = 0
        prev_front_signal = float(np.sum(obs[4:12]) + obs[16])  # Consistent: center + IR

        while not done:
            state_vec = stack_to_state(stack)
            action, log_prob, value = agent.act(state_vec, greedy=False)
            next_obs, env_reward, done = env.step(ACTIONS[action], render=False)

            if action in (0, 1, 3, 4):
                rotate_streak += 1
            else:
                rotate_streak = 0

            expert_action = expert_policy(obs)
            train_reward, prev_front_signal = shaped_reward(
                env_reward, obs, action, rotate_streak, prev_front_signal, ep_step
            )
            ep_step += 1

            memory["states"].append(state_vec)
            memory["actions"].append(action)
            memory["log_probs"].append(log_prob)
            memory["rewards"].append(float(train_reward))
            memory["dones"].append(done)
            memory["values"].append(value)
            memory["expert"].append(expert_action)
            ep_actions.append(action)  # DEBUG: track for diversity check

            ep_raw_total += float(env_reward)
            ep_shaped_total += float(train_reward)

            obs = next_obs
            stack.append(obs.copy())

        memory["total_steps"] = ep + 1  # Track progress for expert probability decay
        loss, pi_loss, v_loss, ent = agent.update(memory, expert_actions=memory["expert"])

        train_hist.append(ep_raw_total)
        shaped_hist.append(ep_shaped_total)
        loss_hist.append(loss)
        pi_hist.append(pi_loss)
        v_hist.append(v_loss)
        ent_hist.append(ent)

        if (ep + 1) % EVAL_EVERY == 0:
            mean_r, std_r = eval_agent(
                build_eval_policy(agent),
                difficulty=DIFFICULTY,
                runs=EVAL_RUNS,
            )

            # DEBUG: action distribution (prevent mode collapse)
            action_counts = np.bincount(ep_actions, minlength=N_ACTIONS)
            action_dist = (action_counts / len(ep_actions) * 100).astype(int)
            
            print(
                f"ep={ep + 1:5d}  train100_raw={float(np.mean(train_hist)):9.1f}  "
                f"train100_shaped={float(np.mean(shaped_hist)):9.1f}  "
                f"eval={mean_r:9.1f}+-{std_r:6.1f}  "
                f"loss={float(np.mean(loss_hist)):.4f}  "
                f"ent={float(np.mean(ent_hist)):.4f}  "
                f"actions={action_dist}"
            )

            logger.log(
                phase=3,
                episode=ep + 1,
                train100=float(np.mean(train_hist)),
                eval_mean=mean_r,
                eval_std=std_r,
                loss=float(np.mean(loss_hist)),
                policy_loss=float(np.mean(pi_hist)),
                value_loss=float(np.mean(v_hist)),
                entropy=float(np.mean(ent_hist)),
                train_shaped=float(np.mean(shaped_hist)),
                difficulty=DIFFICULTY,
            )

            if mean_r > best_eval:
                best_eval = mean_r
                torch.save(agent.model.state_dict(), WEIGHTS_BEST)
                print(f"  New best={best_eval:.1f}; saved {WEIGHTS_BEST}")

    logger.done(best_eval)
    torch.save(agent.model.state_dict(), WEIGHTS_FINAL)
    print(f"Done. Best eval: {best_eval:.1f}")


if __name__ == "__main__":
    train()
