import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_LIST = ["L45", "L22", "FW", "R22", "R45"]
NUM_ACTIONS = 5
OBS_DIM = 38


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=256, n_actions=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, obs):
        feat = self.shared(obs)
        return self.actor(feat), self.critic(feat).squeeze(-1)


def _build_obs(raw, ahist, step, max_steps, in_push, wall_suspect, consec_stuck):
    o = np.array(raw, dtype=np.float32)
    a1 = np.zeros(5, dtype=np.float32)
    a2 = np.zeros(5, dtype=np.float32)
    if len(ahist) >= 1 and ahist[-1] >= 0: a1[ahist[-1]] = 1.0
    if len(ahist) >= 2 and ahist[-2] >= 0: a2[ahist[-2]] = 1.0
    extra = np.array([
        step / max_steps, float(in_push), float(wall_suspect), float(o[17]),
        float(any(o[j] for j in range(17))),
        float(any(o[j] for j in range(4, 12))),
        float(any(o[j] for j in range(0, 4))),
        float(any(o[j] for j in range(12, 16))),
        float(o[16]),
        min(consec_stuck / 10.0, 1.0),
    ], dtype=np.float32)
    return np.concatenate([o, a1, a2, extra])


_model = None
_initialized = False
_step = 0
_in_push = False
_ir_was_on = False
_consec_push = 0
_wall_suspect = False
_consec_stuck = 0
_stuck_dir = 0
_prev_action = -1
_ahist = []
_prev_obs = None
_steps_since_sensor = 0


def _load():
    global _model
    for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth"),
              "weights.pth"]:
        if os.path.exists(p):
            _model = ActorCritic(obs_dim=OBS_DIM, hidden=256)
            _model.load_state_dict(torch.load(p, map_location="cpu"))
            _model.eval()
            print(f"Loaded PPO model from {p}")
            return
    print("WARNING: No weights.pth found, using pure reactive policy")
    _model = None


def _reset():
    global _step, _in_push, _ir_was_on, _consec_push, _wall_suspect
    global _consec_stuck, _stuck_dir, _prev_action, _ahist, _prev_obs, _steps_since_sensor
    _step = 0; _in_push = False; _ir_was_on = False; _consec_push = 0
    _wall_suspect = False; _consec_stuck = 0; _stuck_dir = 0
    _prev_action = -1; _ahist = [-1, -1]; _prev_obs = None; _steps_since_sensor = 0


def _update(obs, action):
    global _in_push, _ir_was_on, _consec_push, _wall_suspect
    global _consec_stuck, _stuck_dir, _steps_since_sensor, _prev_action
    
    ir = bool(obs[16])
    any_near = any(bool(obs[2*i+1]) for i in range(8))
    any_s = any(bool(obs[j]) for j in range(17))
    stuck = bool(obs[17])
    
    if not _in_push:
        if ir and any_near: _in_push = True
        if _ir_was_on and any_s:
            _consec_push += 1
            if _consec_push >= 2: _in_push = True
        elif any_s: _consec_push += 1
        else: _consec_push = 0
    _ir_was_on = ir
    
    if stuck:
        _consec_stuck += 1
        if stuck and _prev_action == 2 and any_s:
            _wall_suspect = True
    else:
        _consec_stuck = 0
    if not any_s and not stuck:
        _wall_suspect = False
    
    if any_s: _steps_since_sensor = 0
    else: _steps_since_sensor += 1
    
    _prev_action = action


def _reactive(obs, rng):
    """Reactive fallback policy."""
    o = np.asarray(obs, dtype=int)
    stuck = bool(o[17]); ir = bool(o[16])
    left_any = bool(o[0] or o[1] or o[2] or o[3])
    front_any = bool(o[4] or o[5] or o[6] or o[7] or o[8] or o[9] or o[10] or o[11])
    right_any = bool(o[12] or o[13] or o[14] or o[15])
    any_s = left_any or front_any or right_any or ir
    
    def rand():
        return float(rng.random()) if rng is not None else float(np.random.random())
    
    global _consec_stuck, _stuck_dir
    
    if stuck:
        cycle = _consec_stuck % 3
        if _consec_stuck > 0 and _consec_stuck % 9 == 0:
            _stuck_dir = 1 - _stuck_dir
        if cycle < 2:
            return 0 if _stuck_dir == 0 else 4
        return 2
    
    if _in_push:
        if ir or front_any: return 2
        if left_any: return 1
        if right_any: return 3
        return 2
    
    if _wall_suspect and any_s and not ir:
        if left_any and not right_any: return 4
        if right_any and not left_any: return 0
        return 0 if (_step % 6 < 3) else 4
    
    if ir: return 2
    if front_any: return 2
    if left_any: return 1
    if right_any: return 3
    
    r = rand()
    if _steps_since_sensor < 40:
        if r < 0.70: return 2
        elif r < 0.82: return 1
        elif r < 0.94: return 3
        elif r < 0.97: return 0
        else: return 4
    elif _steps_since_sensor < 100:
        if r < 0.55: return 2
        elif r < 0.70: return 1
        elif r < 0.85: return 3
        elif r < 0.92: return 0
        else: return 4
    else:
        if r < 0.40: return 2
        elif r < 0.55: return 0
        elif r < 0.70: return 4
        elif r < 0.85: return 1
        else: return 3


def policy(obs, rng=None) -> str:
    global _initialized, _step, _ahist, _prev_obs
    
    if not _initialized:
        _load()
        _reset()
        _initialized = True
    
    obs_arr = np.asarray(obs, dtype=float)
    if _prev_obs is not None and _step > 5:
        if np.sum(np.abs(obs_arr - _prev_obs)) > 8:
            _reset()
    _prev_obs = obs_arr.copy()
    _step += 1
    
    stuck = bool(obs[17])
    any_s = any(bool(obs[j]) for j in range(17))
    
    if stuck:
        a = _reactive(obs, rng)
        _update(obs, a)
        _ahist.append(a)
        if len(_ahist) > 5: _ahist = _ahist[-5:]
        return ACTION_LIST[a]
    
    if not any_s and not _in_push:
        a = _reactive(obs, rng)
        _update(obs, a)
        _ahist.append(a)
        if len(_ahist) > 5: _ahist = _ahist[-5:]
        return ACTION_LIST[a]
    
    if _model is not None:
        aug = _build_obs(obs, _ahist, _step, 2000, _in_push, _wall_suspect, _consec_stuck)
        with torch.no_grad():
            obs_t = torch.FloatTensor(aug).unsqueeze(0)
            logits, _ = _model(obs_t)
            probs = F.softmax(logits, dim=-1)
            a = probs.argmax(dim=-1).item()
        
        if _wall_suspect and any_s and not bool(obs[16]) and a == 2:
            a = _reactive(obs, rng)
    else:
        a = _reactive(obs, rng)
    
    _update(obs, a)
    _ahist.append(a)
    if len(_ahist) > 5: _ahist = _ahist[-5:]
    return ACTION_LIST[a]