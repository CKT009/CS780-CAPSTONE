import random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from obelix_fast import OBELIXFast as OBELIX
from eval import evaluate_agent as _eval_agent
from logger import ExperimentLogger

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_GPU = device.type == "cuda"
print("Device:", device)

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
OBS_DIM   = 18

GAMMA         = 0.99
LR            = 1e-3
BUFFER_CAP    = 50_000
BATCH_SIZE    = 256
TARGET_UPDATE = 200
HIDDEN        = 128
UPDATE_EVERY  = 4 if ON_GPU else 8
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 0.997
WARMUP_EPS    = 10
EVAL_EPISODES = 10
EVAL_SEED     = 88888
EVAL_EVERY    = 100 if ON_GPU else 200

CURRICULUM = [
    dict(arena_size=500, scaling_factor=5, max_steps=1000, wall_obstacles=True, difficulty=3, box_speed=2, episodes=600),
    dict(arena_size=500, scaling_factor=5, max_steps=1000, wall_obstacles=True, difficulty=3, box_speed=2, episodes=1500),
]

HPARAMS = dict(gamma=GAMMA, lr=LR, buffer_cap=BUFFER_CAP, batch_size=BATCH_SIZE,
               target_update=TARGET_UPDATE, hidden=HIDDEN, update_every=UPDATE_EVERY,
               eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,
               seed=SEED, device=str(device))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, obs, act, rew, nobs, done):
        self.buf.append((obs, act, rew, nobs, done))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs),  dtype=torch.float32).to(device),
            torch.tensor(act,            dtype=torch.long).to(device),
            torch.tensor(rew,            dtype=torch.float32).to(device),
            torch.tensor(np.array(nobs), dtype=torch.float32).to(device),
            torch.tensor(done,           dtype=torch.float32).to(device),
        )
    def __len__(self): return len(self.buf)

class DQN(nn.Module):
    def __init__(self, in_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x): return self.net(x)

class Agent:
    def __init__(self):
        self.online = DQN().to(device); self.target = DQN().to(device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=LR)
        self.buf = ReplayBuffer(BUFFER_CAP)
        self.epsilon = EPS_START; self.steps = 0; self.grad_steps = 0
    def act(self, obs, greedy=False):
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            return int(self.online(x).argmax().item())
    def push(self, obs, act, r_raw, nobs, done):
        self.buf.push(obs.astype(np.float32).copy(), act, r_raw,
                      nobs.astype(np.float32).copy(), float(done))
        self.steps += 1
    def update(self):
        if len(self.buf) < BATCH_SIZE or self.steps % UPDATE_EVERY != 0: return None
        obs, act, rew, nobs, done = self.buf.sample(BATCH_SIZE)
        q_curr = self.online(obs).gather(1, act.unsqueeze(1))
        with torch.no_grad():
            q_next = self.target(nobs).max(1)[0].unsqueeze(1)
            q_tgt  = rew.unsqueeze(1) + GAMMA * q_next * (1.0 - done.unsqueeze(1))
        loss = F.huber_loss(q_curr, q_tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
    def decay_eps(self): self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

def evaluate(agent, sf, arena, msteps, difficulty, wall):
    env = OBELIX(scaling_factor=sf, arena_size=arena, max_steps=msteps,
                 wall_obstacles=wall, difficulty=difficulty, box_speed=2, seed=EVAL_SEED)
    scores = []
    for i in range(EVAL_EPISODES):
        obs = env.reset(seed=EVAL_SEED + i); total, done = 0.0, False
        while not done:
            a = agent.act(obs, greedy=True)
            obs, r, done = env.step(ACTIONS[a], render=False)
            total += r
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))

agent  = Agent()
best   = -float("inf")
logger = ExperimentLogger("dqn", HPARAMS)

for phase_idx, phase in enumerate(CURRICULUM):
    arena, sf    = phase["arena_size"], phase["scaling_factor"]
    msteps, wall = phase["max_steps"],  phase["wall_obstacles"]
    diff, n_eps  = phase["difficulty"],  phase["episodes"]

    print(f"\n{'='*60}")
    print(f"Phase {phase_idx+1}: arena={arena} sf={sf} diff={diff} wall={wall} episodes={n_eps}")
    print(f"{'='*60}")

    env = OBELIX(scaling_factor=sf, arena_size=arena, max_steps=msteps,
                 wall_obstacles=wall, difficulty=diff, box_speed=phase["box_speed"])
    if phase_idx > 0: agent.epsilon = max(0.3, agent.epsilon)

    history = deque(maxlen=100); losses = deque(maxlen=200)

    for ep in tqdm(range(n_eps), desc=f"Phase {phase_idx+1}"):
        obs = env.reset(seed=random.randint(0, 1_000_000)); total, done = 0.0, False
        while not done:
            a = agent.act(obs)
            nobs, r, done = env.step(ACTIONS[a], render=False)
            agent.push(obs, a, r, nobs, done)
            loss = agent.update()
            if loss is not None: losses.append(loss)
            total += r; obs = nobs
        if ep >= WARMUP_EPS: agent.decay_eps()
        history.append(total)

        if (ep + 1) % EVAL_EVERY == 0:
            mean_r, std_r = _eval_agent(lambda obs,rng: ACTIONS[agent.act(obs,greedy=True)], difficulty=diff)
            loss_mean = float(np.mean(losses)) if losses else 0.0
            print(f"  ep={ep+1:5d}  train100={np.mean(history):9.1f}  "
                  f"eval={mean_r:9.1f}±{std_r:6.1f}  eps={agent.epsilon:.3f}  loss={loss_mean:.4f}")
            logger.log(phase=phase_idx+1, episode=ep+1,
                       train100=float(np.mean(history)),
                       eval_mean=mean_r, eval_std=std_r,
                       epsilon=agent.epsilon, loss=loss_mean,
                       arena=arena, sf=sf, difficulty=diff)
            if mean_r > best:
                best = mean_r
                torch.save(agent.online.state_dict(), "weights_dqn.pth")
                print(f"    ↑ best={best:.1f}  saved weights_dqn.pth")

logger.done(best)
print(f"\nDone. Best eval: {best:.1f}")
torch.save(agent.online.state_dict(), "weights_dqn_final.pth")