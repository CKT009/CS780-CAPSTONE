"""
PPO Agent for Phase 3.
Uses network to learn handling of boundary cases (stuck flag) instead of hardcoded penalties.
"""
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_STACK = None
_LAST_RNG_ID = None
_STACK_SIZE = 8


def _load():
    global _MODEL
    if _MODEL is not None:
        return

    class PPOActorCritic(nn.Module):
        def __init__(self, in_dim=144, hidden=192, n_actions=5):
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

    model = PPOActorCritic()
    weights_path = os.path.join(os.path.dirname(__file__), "weights_phase3_ppo_diff3.pth")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    _MODEL = model


def _stacked_obs(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    global _STACK, _LAST_RNG_ID

    rng_id = id(rng)
    if _STACK is None or _LAST_RNG_ID != rng_id:
        _LAST_RNG_ID = rng_id
        _STACK = deque(maxlen=_STACK_SIZE)
        for _ in range(_STACK_SIZE):
            _STACK.append(obs.copy())
    else:
        _STACK.append(obs.copy())

    return np.concatenate(_STACK, axis=0)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    PPO policy function.
    Network sees full observation including stuck flag (bit 17).
    Network learns to handle boundary cases through policy gradients.
    """
    _load()
    state = _stacked_obs(obs, rng)
    x = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = _MODEL(x)
        action_idx = int(torch.argmax(logits.squeeze(0)).item())
    return ACTIONS[action_idx]
