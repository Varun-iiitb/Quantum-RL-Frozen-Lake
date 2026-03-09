"""
=============================================================
  Phase 2: DQN (Deep Q-Network) Agent — FrozenLake-v1
=============================================================
  Quantum RL Project — FrozenLake Edition
  Author : Varun E
  Date   : March 2026

  Algorithm: Deep Q-Network (DQN)  [Mnih et al., 2015]

  Key components over tabular Q-Learning:
    1. Neural network function approximator (Q-network)
    2. Experience Replay Buffer — break temporal correlations
    3. Target Network             — stabilize training targets

  Architecture:
    Input(16) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(4)

  Hyperparameters:
    gamma       = 0.99
    lr          = 0.001  (Adam)
    eps         = 1.0 -> 0.01  (x0.995 / episode)
    batch_size  = 64
    buffer_cap  = 10,000
    target_sync = every 100 steps
    min_buffer  = 500  (start training only after this many samples)
    episodes    = 10,000
=============================================================
"""

import sys
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Hyperparameters
# ─────────────────────────────────────────────────────────────

GAMMA          = 0.99        # discount factor
LR             = 0.001       # Adam learning rate
EPS_START      = 1.0         # initial exploration rate
EPS_MIN        = 0.01        # floor for epsilon
EPS_DECAY      = 0.995       # multiplicative decay per episode
BATCH_SIZE     = 64          # replay mini-batch size
BUFFER_CAPACITY= 10_000      # max transitions in replay buffer
TARGET_SYNC    = 100         # hard-update target net every N steps
MIN_BUFFER     = 500         # don't train until this many samples stored
N_EPISODES     = 10_000      # total training episodes
EVAL_EPS       = 1_000       # greedy evaluation episodes
N_STATES       = 16          # FrozenLake 4x4
N_ACTIONS      = 4           # LEFT(0) DOWN(1) RIGHT(2) UP(3)

print("=" * 62)
print("    PHASE 2: DQN AGENT ON FROZENLAKE-v1")
print("=" * 62)
print(f"\n  Hyperparameters:")
print(f"    gamma              = {GAMMA}")
print(f"    learning rate      = {LR}  (Adam optimizer)")
print(f"    epsilon start/min  = {EPS_START} -> {EPS_MIN}")
print(f"    epsilon decay      = {EPS_DECAY} per episode")
print(f"    batch size         = {BATCH_SIZE}")
print(f"    replay buffer cap  = {BUFFER_CAPACITY:,}")
print(f"    target net sync    = every {TARGET_SYNC} steps")
print(f"    min buffer to train= {MIN_BUFFER}")
print(f"    training episodes  = {N_EPISODES:,}")
print(f"    eval episodes      = {EVAL_EPS:,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: State Encoding — One-Hot
# ─────────────────────────────────────────────────────────────
# FrozenLake states are integers 0-15.
# DQN needs a continuous vector input, so we one-hot encode:
#   state 5  ->  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# This gives the network a structured 16-dim input per state.

def one_hot(state: int, n_states: int = N_STATES) -> torch.Tensor:
    """Convert discrete state integer to one-hot float tensor."""
    vec = torch.zeros(n_states, dtype=torch.float32, device=device)
    vec[state] = 1.0
    return vec

# ─────────────────────────────────────────────────────────────
# SECTION 3: Q-Network Architecture
# ─────────────────────────────────────────────────────────────
# A simple 3-layer MLP: Input(16) -> 64 -> 64 -> 4
# Output: Q-value estimates for all 4 actions simultaneously.
# Using ReLU activations — standard for DQN-style networks.

class QNetwork(nn.Module):
    """
    Deep Q-Network: maps state-vector -> Q-values for each action.

    Architecture:
        Linear(16 -> 64) -> ReLU
        Linear(64 -> 64) -> ReLU
        Linear(64 -> 4)            <- Q(s, each action)
    """
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Instantiate online Q-network and target Q-network
# The online network is updated every step; the target network
# provides stable bootstrap targets and is synced every 100 steps.
q_net      = QNetwork(N_STATES, N_ACTIONS).to(device)
target_net = QNetwork(N_STATES, N_ACTIONS).to(device)

# Initialize target net with the same weights as the online net
target_net.load_state_dict(q_net.state_dict())
target_net.eval()   # target net is never directly trained

optimizer = optim.Adam(q_net.parameters(), lr=LR)
loss_fn   = nn.MSELoss()

total_params = sum(p.numel() for p in q_net.parameters())
print(f"\n  Q-Network architecture:")
print(q_net)
print(f"  Total trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Experience Replay Buffer
# ─────────────────────────────────────────────────────────────
# WHY needed: In naive DQN without replay, each (s,a,r,s') tuple is
# used only once and discarded — very sample inefficient.
# Consecutive transitions are also highly correlated, making the
# gradient updates noisy and unstable.
#
# The replay buffer stores up to BUFFER_CAPACITY transitions.
# At each training step we sample a random MINI-BATCH, breaking
# temporal correlations and allowing the same transitions to be
# reused many times.

Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)

class ReplayBuffer:
    """
    Circular experience replay buffer.
    Stores (state, action, reward, next_state, done) transitions.
    """
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a new transition to the buffer (auto-evicts oldest if full)."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Randomly sample a mini-batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
print(f"\n  Replay buffer initialized  (capacity={BUFFER_CAPACITY:,})")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Training Helper — Sample & Update
# ─────────────────────────────────────────────────────────────

def train_step() -> float:
    """
    Sample a mini-batch from replay buffer and perform one
    gradient descent step on the online Q-network.

    Loss = MSE( Q(s,a),  r + gamma * max_a' Q_target(s') * (1-done) )
    """
    batch       = replay_buffer.sample(BATCH_SIZE)
    states      = torch.stack([t.state      for t in batch])          # [B, 16]
    actions     = torch.tensor([t.action    for t in batch],
                               dtype=torch.long, device=device)        # [B]
    rewards     = torch.tensor([t.reward    for t in batch],
                               dtype=torch.float32, device=device)     # [B]
    next_states = torch.stack([t.next_state for t in batch])          # [B, 16]
    dones       = torch.tensor([t.done      for t in batch],
                               dtype=torch.float32, device=device)     # [B]

    # Current Q-values: Q(s, a) for each sample in the batch
    # We select only the Q-value corresponding to the taken action
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

    # Target Q-values using the TARGET network (not the online net)
    # WHY target net: if we used the online net for both prediction
    # and target, the targets would shift every update -> instability.
    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1)[0]             # [B]
        targets    = rewards + GAMMA * max_next_q * (1.0 - dones)      # [B]

    # Compute MSE loss and backprop
    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping — prevents exploding gradients
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

# ─────────────────────────────────────────────────────────────
# SECTION 6: Main Training Loop
# ─────────────────────────────────────────────────────────────

env             = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
epsilon         = EPS_START
episode_rewards = []    # total reward per episode
episode_success = []    # True/False per episode
epsilon_history = []    # epsilon at start of each episode
total_steps     = 0     # global step counter for target network sync
training_started= False

print("\n" + "=" * 62)
print("              TRAINING PROGRESS")
print("=" * 62)
print(f"  {'Episode':>10}  {'Successes':>12}  {'Success%':>10}  {'Epsilon':>10}  {'Buffer':>8}")
print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")

for episode in range(N_EPISODES):

    state, _ = env.reset()
    ep_reward = 0
    done      = False
    truncated = False

    epsilon_history.append(epsilon)      # record before decay

    while not done and not truncated:
        total_steps += 1

        # ── Epsilon-Greedy Action Selection ──
        if random.random() < epsilon:
            action = env.action_space.sample()      # explore
        else:
            with torch.no_grad():
                q_vals = q_net(one_hot(state))
            action = q_vals.argmax().item()          # exploit

        # ── Environment Step ──
        next_state, reward, done, truncated, _ = env.step(action)

        # ── Store transition in replay buffer ──
        replay_buffer.push(
            one_hot(state),
            action,
            reward,
            one_hot(next_state),
            float(done)
        )

        # ── Train online network (only if buffer has enough samples) ──
        if len(replay_buffer) >= MIN_BUFFER:
            if not training_started:
                print(f"  [Episode {episode+1:,}] Buffer reached {MIN_BUFFER} — training begins!")
                training_started = True
            train_step()

        # ── Hard-update target network every TARGET_SYNC steps ──
        # WHY hard update: periodically copy online weights to target.
        # This freezes the target for TARGET_SYNC steps, giving stable
        # bootstrap values during the interval.
        if total_steps % TARGET_SYNC == 0:
            target_net.load_state_dict(q_net.state_dict())

        ep_reward += reward
        state      = next_state

    # ── Epsilon decay ──
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

    # ── Record episode outcome ──
    success = done and ep_reward > 0
    episode_rewards.append(ep_reward)
    episode_success.append(success)

    # ── Print progress every 1000 episodes ──
    if (episode + 1) % 1000 == 0:
        window_successes = sum(episode_success[-1000:])
        window_rate      = window_successes / 1000 * 100
        print(
            f"  {episode+1:>10,}  "
            f"{window_successes:>12}  "
            f"{window_rate:>9.1f}%  "
            f"{epsilon:>10.4f}  "
            f"{len(replay_buffer):>8,}"
        )

env.close()
print(f"\n  Training complete! {N_EPISODES:,} episodes finished.")

# ─────────────────────────────────────────────────────────────
# SECTION 7: Greedy Policy Evaluation
# ─────────────────────────────────────────────────────────────

eval_env     = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
eval_success = 0
eval_rewards = []

print("\n" + "=" * 62)
print(f"   GREEDY POLICY EVALUATION ({EVAL_EPS:,} episodes, eps=0)")
print("=" * 62)

q_net.eval()   # switch off dropout / batchnorm if any
with torch.no_grad():
    for ep in range(EVAL_EPS):
        state, _ = eval_env.reset(seed=ep)
        done      = False
        truncated = False
        ep_reward = 0

        while not done and not truncated:
            q_vals = q_net(one_hot(state))
            action = q_vals.argmax().item()          # pure greedy
            state, reward, done, truncated, _ = eval_env.step(action)
            ep_reward += reward

        eval_rewards.append(ep_reward)
        if ep_reward > 0:
            eval_success += 1

eval_env.close()
q_net.train()

dqn_success_rate = eval_success / EVAL_EPS * 100
dqn_avg_reward   = np.mean(eval_rewards)

print(f"\n  Successful episodes : {eval_success} / {EVAL_EPS:,}")
print(f"  Success rate        : {dqn_success_rate:.2f}%")
print(f"  Avg reward          : {dqn_avg_reward:.4f}")

# ─────────────────────────────────────────────────────────────
# SECTION 8: Save Model Weights
# ─────────────────────────────────────────────────────────────

torch.save(q_net.state_dict(), "dqn_model.pth")
print(f"\n  Model weights saved --> dqn_model.pth")

# ─────────────────────────────────────────────────────────────
# SECTION 9: Plots — Reward Curve + Epsilon Decay
# ─────────────────────────────────────────────────────────────

print("\n  Generating plots...")

rewards_arr = np.array(episode_rewards, dtype=float)
window      = 100
smoothed    = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
smooth_x    = np.arange(window - 1, N_EPISODES)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "DQN on FrozenLake-v1  (gamma=0.99, lr=0.001, replay buffer, target net)",
    fontsize=12, fontweight="bold", y=1.02
)

# ── Graph 1: Reward Curve ──
ax1 = axes[0]
ax1.plot(rewards_arr, color="#f4a261", alpha=0.3, linewidth=0.5,
         label="Raw episode reward")
ax1.plot(smooth_x, smoothed, color="#e76f51", linewidth=2.0,
         label=f"{window}-ep rolling average")
ax1.axhline(y=dqn_success_rate / 100, color="#264653", linewidth=1.5,
            linestyle="--",
            label=f"Greedy eval: {dqn_success_rate:.1f}%")
# Also draw Q-Learning baseline for direct comparison
ax1.axhline(y=0.723, color="#2a9d8f", linewidth=1.2,
            linestyle=":", alpha=0.8,
            label="Q-Learning eval: 72.3%")

ax1.set_title("DQN — Training Reward Curve", fontsize=11, fontweight="bold")
ax1.set_xlabel("Episode", fontsize=10)
ax1.set_ylabel("Reward (0=fail, 1=success)", fontsize=10)
ax1.set_ylim(-0.05, 1.25)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.3)

# Annotate final smoothed value
final_smooth = smoothed[-1]
ax1.annotate(
    f"Final avg: {final_smooth:.3f}",
    xy=(smooth_x[-1], final_smooth),
    xytext=(smooth_x[-1] - 1800, final_smooth + 0.1),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color="#e76f51"),
    color="#e76f51"
)

# ── Graph 2: Epsilon Decay ──
ax2 = axes[1]
ax2.plot(epsilon_history, color="#e9c46a", linewidth=1.8,
         label="Epsilon (exploration rate)")
ax2.axhline(y=EPS_MIN, color="#e76f51", linewidth=1.2, linestyle="--",
            label=f"Minimum epsilon = {EPS_MIN}")

crossover = next((i for i, e in enumerate(epsilon_history) if e <= 0.5), None)
if crossover:
    ax2.axvline(x=crossover, color="#9467bd", linewidth=1.0,
                linestyle=":", alpha=0.8)
    ax2.text(crossover + 50, 0.52, f"eps=0.5\n@ep {crossover:,}",
             fontsize=7.5, color="#9467bd")

# Shade regions
ax2.fill_between(range(N_EPISODES), epsilon_history, EPS_MIN,
                 where=[e > 0.5 for e in epsilon_history],
                 alpha=0.15, color="#e9c46a", label="Exploration dominant")
ax2.fill_between(range(N_EPISODES), epsilon_history, EPS_MIN,
                 where=[e <= 0.5 for e in epsilon_history],
                 alpha=0.15, color="#264653", label="Exploitation dominant")

ax2.set_title("DQN — Epsilon Decay Over Training", fontsize=11, fontweight="bold")
ax2.set_xlabel("Episode", fontsize=10)
ax2.set_ylabel("Epsilon", fontsize=10)
ax2.set_ylim(-0.02, 1.05)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("dqn_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved --> dqn_results.png")

# ─────────────────────────────────────────────────────────────
# SECTION 10: Comparison Table — Q-Learning vs DQN
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("      ALGORITHM COMPARISON: Q-Learning vs DQN")
print("=" * 62)

headers = f"  {'Metric':<30}  {'Q-Learning':>12}  {'DQN':>10}"
divider = "  " + "-" * 56
print(headers)
print(divider)

rows = [
    ("Algorithm type",          "Tabular",        "Neural Net"),
    ("State representation",    "Integer (0-15)", "One-hot (16-d)"),
    ("Function approximator",   "Q-table [16x4]", "MLP 16->64->64->4"),
    ("Experience replay",       "No",             "Yes (10k buffer)"),
    ("Target network",          "No",             "Yes (sync/100 steps)"),
    ("Optimizer",               "Bellman (manual)","Adam lr=0.001"),
    ("gamma (discount)",        "0.95",           "0.99"),
    ("Training episodes",       "10,000",         "10,000"),
    ("Greedy eval episodes",    "1,000",          "1,000"),
    ("Success rate (eval)",     "72.3%",          f"{dqn_success_rate:.1f}%"),
    ("Avg reward (eval)",       "0.7230",         f"{dqn_avg_reward:.4f}"),
    ("Model size",              "64 values",      f"{total_params:,} params"),
]

for metric, ql_val, dqn_val in rows:
    print(f"  {metric:<30}  {ql_val:>12}  {dqn_val:>10}")

print(divider)
print()

# Performance verdict
if dqn_success_rate > 72.3:
    verdict = (f"DQN outperforms Q-Learning by "
               f"{dqn_success_rate - 72.3:+.1f}% success rate.")
elif abs(dqn_success_rate - 72.3) < 3.0:
    verdict = ("DQN and Q-Learning perform similarly on FrozenLake-v1.\n"
               "  This is expected: tabular methods can be optimal on\n"
               "  small discrete environments (16 states), whereas DQN\n"
               "  shines on large/continuous state spaces.")
else:
    verdict = (f"Q-Learning outperforms DQN by "
               f"{72.3 - dqn_success_rate:+.1f}% here. Tabular methods\n"
               "  can beat DQN on very small discrete spaces due to\n"
               "  exact Q-value storage rather than neural approximation.")

print(f"  Verdict: {verdict}")
print()

print("=" * 62)
print("  DQN training complete. Comparison established.")
print("  Saved: dqn_model.pth  |  dqn_results.png")
print("=" * 62)
