"""
=============================================================
  Phase 2: Q-Learning (Tabular Baseline) — FrozenLake-v1
=============================================================
  Quantum RL Project — FrozenLake Edition
  Author : Varun E
  Date   : March 2026

  Algorithm : Tabular Q-Learning
  Update rule:
      Q[s,a] <- Q[s,a] + alpha * (r + gamma * max(Q[s',:]) - Q[s,a])

  Hyperparameters:
      alpha  = 0.8     (learning rate)
      gamma  = 0.95    (discount factor)
      eps    = 1.0     (initial epsilon for epsilon-greedy)
      eps_decay = 0.995 per episode, min eps = 0.01
      episodes = 10,000
=============================================================
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — works headlessly
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym

# Force UTF-8 so special chars print cleanly on Windows
sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Hyperparameters & Environment Setup
# ─────────────────────────────────────────────────────────────

ALPHA      = 0.8        # learning rate
GAMMA      = 0.95       # discount factor (future reward importance)
EPS_START  = 1.0        # initial exploration rate
EPS_MIN    = 0.01       # minimum exploration rate (never fully greedy)
EPS_DECAY  = 0.995      # multiplicative decay per episode
N_EPISODES = 10_000     # total training episodes
EVAL_EPS   = 1_000      # evaluation episodes (greedy, no exploration)

N_STATES   = 16         # FrozenLake 4x4 grid
N_ACTIONS  = 4          # LEFT=0, DOWN=1, RIGHT=2, UP=3
ACTION_NAMES = ["LEFT ", "DOWN ", "RIGHT", "UP   "]

print("=" * 62)
print("      PHASE 2: Q-LEARNING ON FROZENLAKE-v1")
print("=" * 62)
print(f"\n  Hyperparameters:")
print(f"    alpha (learning rate)   = {ALPHA}")
print(f"    gamma (discount)        = {GAMMA}")
print(f"    epsilon start           = {EPS_START}")
print(f"    epsilon min             = {EPS_MIN}")
print(f"    epsilon decay / episode = {EPS_DECAY}")
print(f"    training episodes       = {N_EPISODES:,}")
print(f"    evaluation episodes     = {EVAL_EPS:,}")

# Initialize training environment (no render — faster)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# ─────────────────────────────────────────────────────────────
# SECTION 2: Q-Table Initialization
# ─────────────────────────────────────────────────────────────
# Q-table shape: [num_states x num_actions] = [16 x 4]
# All values start at 0 — the agent has no prior knowledge.
Q = np.zeros((N_STATES, N_ACTIONS))

print(f"\n  Q-table initialized: shape = {Q.shape}  (all zeros)")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Training Loop — Q-Learning with Epsilon-Greedy
# ─────────────────────────────────────────────────────────────
episode_rewards  = []   # total reward for each training episode
episode_success  = []   # True/False for each episode
epsilon_history  = []   # epsilon value at start of each episode
epsilon          = EPS_START

print("\n" + "=" * 62)
print("              TRAINING PROGRESS")
print("=" * 62)
print(f"  {'Episode':>10}  {'Successes':>12}  {'Success%':>10}  {'Epsilon':>10}")
print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")

for episode in range(N_EPISODES):

    # Reset environment for a new episode
    state, _ = env.reset()
    total_reward = 0
    done         = False
    truncated    = False

    # Record epsilon at the start of this episode (for the plot)
    epsilon_history.append(epsilon)

    while not done and not truncated:

        # ── Epsilon-Greedy Action Selection ──
        # With prob epsilon  → explore: pick a random action
        # With prob 1-epsilon → exploit: pick best Q-value action
        if np.random.random() < epsilon:
            action = env.action_space.sample()          # explore
        else:
            action = np.argmax(Q[state, :])             # exploit

        # ── Take action, observe next state and reward ──
        next_state, reward, done, truncated, _ = env.step(action)

        # ── Bellman Update ──
        # Q[s,a] <- Q[s,a] + alpha * (r + gamma * max_a'(Q[s',:]) - Q[s,a])
        best_next_q    = np.max(Q[next_state, :])
        td_target      = reward + GAMMA * best_next_q * (not done)
        td_error       = td_target - Q[state, action]
        Q[state, action] += ALPHA * td_error

        total_reward += reward
        state         = next_state

    # ── Epsilon Decay ──
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

    # ── Record episode outcome ──
    success = done and total_reward > 0
    episode_rewards.append(total_reward)
    episode_success.append(success)

    # ── Print progress every 1000 episodes ──
    if (episode + 1) % 1000 == 0:
        window_successes = sum(episode_success[-1000:])
        window_rate      = window_successes / 1000 * 100
        print(
            f"  {episode+1:>10,}  "
            f"{window_successes:>12}  "
            f"{window_rate:>9.1f}%  "
            f"{epsilon:>10.4f}"
        )

print(f"\n  Training complete! {N_EPISODES:,} episodes finished.")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Print Final Q-Table
# ─────────────────────────────────────────────────────────────
# Each row = a state (0–15), each column = an action value.
# The best action per state is marked with *.

print("\n" + "=" * 62)
print("         FINAL Q-TABLE  (16 states x 4 actions)")
print("=" * 62)

# Map state index to (row, col) in the 4x4 grid
MAP_CELLS = list("SFFFHFHFFFFHHHFG")  # standard 4x4 FrozenLake map (flattened)

print(f"\n  {'State':>6}  {'Grid':>5}  {'Cell':>4}  |  "
      f"{'LEFT':>8}  {'DOWN':>8}  {'RIGHT':>8}  {'UP':>8}  |  Best Action")
print(f"  {'------':>6}  {'-----':>5}  {'----':>4}  |  "
      f"{'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  |  -----------")

for s in range(N_STATES):
    row, col   = s // 4, s % 4
    cell_type  = MAP_CELLS[s]
    best_a     = np.argmax(Q[s, :])
    best_name  = ACTION_NAMES[best_a].strip()

    # Mark Holes and Goal specially
    if cell_type in ("H", "G"):
        best_str = f"({cell_type})"
    else:
        best_str = best_name

    q_vals = "  ".join(f"{Q[s, a]:>8.4f}" for a in range(N_ACTIONS))
    print(
        f"  {s:>6}  ({row},{col})  [{cell_type}]  |  "
        f"{q_vals}  |  {best_str}"
    )

print()

# ─────────────────────────────────────────────────────────────
# SECTION 5: Greedy Policy Evaluation (1000 Episodes)
# ─────────────────────────────────────────────────────────────
# After training, we evaluate with epsilon=0 (pure exploitation).
# This gives a clean estimate of how good the learned policy is.

eval_env        = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
eval_success    = 0
eval_rewards    = []

print("=" * 62)
print(f"   GREEDY POLICY EVALUATION ({EVAL_EPS:,} episodes, eps=0)")
print("=" * 62)

for ep in range(EVAL_EPS):
    state, _ = eval_env.reset(seed=ep)
    done      = False
    truncated = False
    ep_reward = 0

    while not done and not truncated:
        action = np.argmax(Q[state, :])         # always pick best Q action
        state, reward, done, truncated, _ = eval_env.step(action)
        ep_reward += reward

    eval_rewards.append(ep_reward)
    if ep_reward > 0:
        eval_success += 1

eval_env.close()

eval_success_rate = eval_success / EVAL_EPS * 100
eval_avg_reward   = np.mean(eval_rewards)
print(f"\n  Successful episodes : {eval_success} / {EVAL_EPS:,}")
print(f"  Success rate        : {eval_success_rate:.2f}%")
print(f"  Avg reward          : {eval_avg_reward:.4f}")
print()
print(f"  Note: Random baseline success rate was ~20%")
print(f"  Q-Learning achieved {eval_success_rate:.1f}% -- "
      f"{'a significant improvement!' if eval_success_rate > 20 else 'see analysis below.'}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Save Q-Table
# ─────────────────────────────────────────────────────────────
# Save for later comparison with Policy Gradient, DQN, etc.
np.save("q_table.npy", Q)
print(f"\n  Q-table saved to  --> q_table.npy")
print(f"  Shape: {Q.shape}, dtype: {Q.dtype}")

# ─────────────────────────────────────────────────────────────
# SECTION 7: Plots — Reward Curve + Epsilon Decay
# ─────────────────────────────────────────────────────────────
# Graph 1: Raw per-episode rewards + 100-episode rolling average
# Graph 2: Epsilon decay over all training episodes

print("\n  Generating plots...")

# ── Smoothed reward (100-episode rolling average) ──
rewards_arr = np.array(episode_rewards, dtype=float)
window       = 100
smoothed     = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
smooth_x     = np.arange(window - 1, N_EPISODES)   # x-axis for smoothed curve

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Q-Learning on FrozenLake-v1  (alpha=0.8, gamma=0.95, 10,000 episodes)",
    fontsize=13, fontweight="bold", y=1.02
)

# ── Graph 1: Reward Curve ──
ax1 = axes[0]
ax1.plot(
    rewards_arr, color="#b0c4de", alpha=0.35, linewidth=0.6,
    label="Raw episode reward"
)
ax1.plot(
    smooth_x, smoothed, color="#1f77b4", linewidth=2.0,
    label=f"{window}-episode rolling average"
)
# Mark the evaluated success rate as a horizontal reference line
ax1.axhline(
    y=eval_success_rate / 100, color="#d62728",
    linewidth=1.5, linestyle="--",
    label=f"Greedy eval success rate: {eval_success_rate:.1f}%"
)

ax1.set_title("Training Reward Curve", fontsize=11, fontweight="bold")
ax1.set_xlabel("Episode", fontsize=10)
ax1.set_ylabel("Reward (0 = failure, 1 = success)", fontsize=10)
ax1.set_ylim(-0.05, 1.15)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.3)

# Annotate final smoothed reward
final_smooth = smoothed[-1]
ax1.annotate(
    f"Final avg: {final_smooth:.3f}",
    xy=(smooth_x[-1], final_smooth),
    xytext=(smooth_x[-1] - 1500, final_smooth + 0.08),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color="#1f77b4"),
    color="#1f77b4"
)

# ── Graph 2: Epsilon Decay ──
ax2 = axes[1]
ax2.plot(
    epsilon_history, color="#2ca02c", linewidth=1.8,
    label="Epsilon (exploration rate)"
)
ax2.axhline(y=EPS_MIN, color="#ff7f0e", linewidth=1.2, linestyle="--",
            label=f"Minimum epsilon = {EPS_MIN}")

# Shade the "mostly exploring" vs "mostly exploiting" regions
ax2.fill_between(
    range(N_EPISODES), epsilon_history, EPS_MIN,
    where=[e > 0.5 for e in epsilon_history],
    alpha=0.12, color="#2ca02c", label="Exploration dominant (eps > 0.5)"
)
ax2.fill_between(
    range(N_EPISODES), epsilon_history, EPS_MIN,
    where=[e <= 0.5 for e in epsilon_history],
    alpha=0.12, color="#1f77b4", label="Exploitation dominant (eps <= 0.5)"
)

# Mark roughly where epsilon crosses 0.5
crossover = next((i for i, e in enumerate(epsilon_history) if e <= 0.5), None)
if crossover:
    ax2.axvline(x=crossover, color="#9467bd", linewidth=1.0,
                linestyle=":", alpha=0.8)
    ax2.text(crossover + 50, 0.52, f"eps=0.5\n@ep {crossover:,}",
             fontsize=7.5, color="#9467bd")

ax2.set_title("Epsilon Decay Over Training", fontsize=11, fontweight="bold")
ax2.set_xlabel("Episode", fontsize=10)
ax2.set_ylabel("Epsilon", fontsize=10)
ax2.set_ylim(-0.02, 1.05)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("q_learning_results.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Plot saved   --> q_learning_results.png")

# ─────────────────────────────────────────────────────────────
# SECTION 8: Final Summary Print
# ─────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("             FINAL SUMMARY")
print("=" * 62)
print(f"\n  Algorithm            : Tabular Q-Learning")
print(f"  Environment          : FrozenLake-v1 (4x4, is_slippery=True)")
print(f"  Training episodes    : {N_EPISODES:,}")
print(f"  alpha (lr)           : {ALPHA}")
print(f"  gamma (discount)     : {GAMMA}")
print(f"  Epsilon decay        : {EPS_DECAY} per episode  ({EPS_START} -> {EPS_MIN})")
print()
print(f"  Training Results (last 1000 episodes):")
train_last_1k = sum(episode_success[-1000:])
print(f"    Successes          : {train_last_1k} / 1000")
print(f"    Success rate       : {train_last_1k / 10:.1f}%")
print()
print(f"  Evaluation Results ({EVAL_EPS:,} greedy episodes):")
print(f"    Successes          : {eval_success} / {EVAL_EPS:,}")
print(f"    Success rate       : {eval_success_rate:.2f}%")
print(f"    Avg reward         : {eval_avg_reward:.4f}")
print()
print(f"  Saved files:")
print(f"    q_table.npy              -- learned Q-values for later comparison")
print(f"    q_learning_results.png   -- reward curve + epsilon decay plot")
print()
print("=" * 62)
print("  Q-Learning complete! Baseline established for comparison.")
print("  Next: Policy Gradient / DQN / PPO / Quantum RL methods.")
print("=" * 62)

env.close()
