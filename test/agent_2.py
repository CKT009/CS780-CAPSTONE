"""
OBELIX Agent — Tabular Q(lambda) Policy
========================================
Codabench submission file.
Submit: agent.py + weights.json inside submission.zip

This agent:
1. Loads pre-trained Q-tables from weights.json
2. Compresses 18-bit obs → small state tuple
3. Detects find/push mode from sensor history
4. Selects action greedily from appropriate Q-table
5. Falls back to hand-coded reactive policy if state unseen

NO PyTorch. Pure numpy + stdlib. Runs on CPU trivially.
"""

import os
import json
import numpy as np
from collections import defaultdict

ACTION_LIST = ["L45", "L22", "FW", "R22", "R45"]
NUM_ACTIONS = 5


def _compress(obs):
    """Compress 18 bits → (left, front_left, front_right, right, ir, stuck)"""
    o = np.asarray(obs, dtype=int)
    
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



def _reactive_action(obs, in_push, step_count):
    """
    Rule-based policy for states not in Q-table.
    Based on the original Mahadevan & Connell (1991) approach.
    """
    o = np.asarray(obs, dtype=int)
    stuck = bool(o[17])
    ir = bool(o[16])
    
    left_any = bool(o[0] or o[1] or o[2] or o[3])
    front_any = bool(o[4] or o[5] or o[6] or o[7] or o[8] or o[9] or o[10] or o[11])
    right_any = bool(o[12] or o[13] or o[14] or o[15])
    
    left_near = bool(o[1] or o[3])
    front_near = bool(o[5] or o[7] or o[9] or o[11])
    right_near = bool(o[13] or o[15])
    
    any_sensor = left_any or front_any or right_any or ir
    
    if stuck:
        if step_count % 4 < 2:
            return 0  # L45
        else:
            return 4  # R45
    
    if ir:
        return 2  
    
    # PUSH MODE
    if in_push:
        if front_near:
            return 2  
        elif front_any:
            return 2  
        elif left_any:
            return 1  
        elif right_any:
            return 3  
        else:
            return 2  
    
    # FIND MODE
    if front_near:
        return 2  
    if front_any:
        return 2 
    if left_near:
        return 0 
    if left_any:
        return 1  
    if right_near:
        return 4  
    if right_any:
        return 3 
    
    pattern = step_count % 20
    if pattern < 13:
        return 2 
    elif pattern < 16:
        return 1 
    elif pattern < 19:
        return 3 
    else:
        return 0 


# ──────────────────────────────────────────────
# Global agent state
# ──────────────────────────────────────────────
_q_find = None
_q_push = None
_initialized = False
_in_push = False
_ir_was_on = False
_consecutive_sensor = 0
_step_count = 0
_prev_obs = None
_stuck_turn_dir = 0
_stuck_count = 0


def _load_tables():
    global _q_find, _q_push
    
    paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json"),
        "weights.json",
    ]
    
    _q_find = {}
    _q_push = {}
    
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
            
            for k, v in data.get("q_find", {}).items():
                _q_find[eval(k)] = np.array(v)
            for k, v in data.get("q_push", {}).items():
                _q_push[eval(k)] = np.array(v)
            
            print(f"Loaded Q-tables: find={len(_q_find)}, push={len(_q_push)}")
            return
    
    print("WARNING: weights.json not found, using pure reactive policy")


def _detect_mode(obs):
    global _in_push, _ir_was_on, _consecutive_sensor
    
    ir = bool(obs[16])
    any_near = any(bool(obs[2*i+1]) for i in range(8))
    any_sensor = any(bool(obs[j]) for j in range(17))
    
    if not _in_push:
        if _ir_was_on and any_sensor:
            _consecutive_sensor += 1
            if _consecutive_sensor >= 2:
                _in_push = True
        elif any_sensor:
            _consecutive_sensor += 1
        else:
            _consecutive_sensor = 0
        
        if ir and any_near:
            _in_push = True
    
    _ir_was_on = ir


def policy(obs) -> str:
    
    global _initialized, _step_count, _prev_obs, _in_push
    global _ir_was_on, _consecutive_sensor, _stuck_count, _stuck_turn_dir
    
    if not _initialized:
        _load_tables()
        _initialized = True
        _in_push = False
        _ir_was_on = False
        _consecutive_sensor = 0
        _step_count = 0
        _prev_obs = None
        _stuck_count = 0
        _stuck_turn_dir = 0
    
    obs_arr = np.asarray(obs, dtype=float)
    if _prev_obs is not None and _step_count > 10:
        diff = np.sum(np.abs(obs_arr - _prev_obs))
        if diff > 8:
            _in_push = False
            _ir_was_on = False
            _consecutive_sensor = 0
            _step_count = 0
            _stuck_count = 0
    
    _step_count += 1
    _detect_mode(obs)
    
    stuck = bool(obs[17])
    if stuck:
        _stuck_count += 1
        if _stuck_count >= 3:
            _stuck_turn_dir = 1 - _stuck_turn_dir
            _stuck_count = 0
    else:
        _stuck_count = 0
    
    state = _compress(obs)
    
    q_table = _q_push if _in_push else _q_find
    
    if q_table is not None and state in q_table:
        q_vals = q_table[state]
        max_q = np.max(q_vals)
        max_actions = np.where(np.abs(q_vals - max_q) < 1e-8)[0]
        
        if len(max_actions) > 1:
            priority = {2: 0, 1: 1, 3: 1, 0: 2, 4: 2}
            action_idx = int(min(max_actions, key=lambda a: priority.get(a, 3)))
        else:
            action_idx = int(max_actions[0])
        
        if stuck and action_idx == 2:
            if _stuck_turn_dir == 0:
                action_idx = 0 
            else:
                action_idx = 4 
    else:
        action_idx = _reactive_action(obs, _in_push, _step_count)
    
    _prev_obs = obs_arr.copy()
    return ACTION_LIST[action_idx]