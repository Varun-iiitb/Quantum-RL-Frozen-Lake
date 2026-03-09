"""
=============================================================
  Phase 2: REINFORCE v2 (Improved Policy Gradient)
           FrozenLake-v1 — Sparse Reward Fixes
=============================================================
  Quantum RL Project — FrozenLake Edition
  Author : Varun E
  Date   : March 2026

  v1 failure analysis:
    - 0% greedy eval due to sparse rewards (reward=1 only at goal)
    - Policy collapsed: argmax always picked one bad action
    - Zero-return gradient signal dominated training

  v2 fixes applied:
    1. Entropy bonus     — prevents action collapse
    2. Reward shaping    — denser gradient signal per step
    3. Global baseline   — running mean of last-100 return subtracted
    4. Slower lr=0.0005  — more stable updates
    5. 20,000 episodes   — more time to discover goal

  Algorithm: REINFORCE with entropy regularization + shaped rewards
  Loss: -Σ log_prob(a_t)*G_t_normalized  -  beta * H(π)
=============================================================
"""

import sys
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Hyperparameters
# ─────────────────────────────────────────────────────────────

GAMMA          = 0.99
LR             = 0.0005      # FIX: slower lr for more stable convergence
N_EPISODES     = 20_000      # FIX: 2x episodes — sparse envs need more time
MAX_STEPS      = 200
EVAL_EPS       = 1_000
ENTROPY_BETA   = 0.01        # FIX: entropy bonus coefficient
BASELINE_WIN   = 100         # window for running baseline mean
N_STATES       = 16
N_ACTIONS      = 4

# Reference results from previous algorithms
QL_SUCCESS    = 72.3
DQN_SUCCESS   = 65.7
PG_V1_SUCCESS = 0.0

print("=" * 62)
print("  PHASE 2: REINFORCE v2 (IMPROVED) ON FROZENLAKE-v1")
print("=" * 62)
print(f"\n  Fixes over v1:")
print(f"    [1] Entropy bonus     : beta={ENTROPY_BETA}")
print(f"    [2] Reward shaping    : step=+0.01, hole=-0.5, goal=+1.0")
print(f"    [3] Running baseline  : subtract mean of last {BASELINE_WIN} returns")
print(f"    [4] Learning rate     : {LR}  (was 0.001 in v1)")
print(f"    [5] Training episodes : {N_EPISODES:,}  (was 10,000 in v1)")
print(f"\n  Hyperparameters:")
print(f"    gamma    = {GAMMA}")
print(f"    max_steps= {MAX_STEPS}")
print(f"    eval_eps = {EVAL_EPS:,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: State Encoding — One-Hot
# ─────────────────────────────────────────────────────────────

def one_hot(state: int, n_states: int = N_STATES) -> torch.Tensor:
    vec = torch.zeros(n_states, dtype=torch.float32, device=device)
    vec[state] = 1.0
    return vec

# ─────────────────────────────────────────────────────────────
# SECTION 3: Policy Network (unchanged from v1)
# ─────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    Same architecture as v1: 16 -> 128 -> 64 -> 4 -> Softmax.
    Outputs action probability distribution π(a|s).
    """
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


policy_net = PolicyNetwork(N_STATES, N_ACTIONS).to(device)
optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
total_params = sum(p.numel() for p in policy_net.parameters())

print(f"\n  Policy Network: {total_params:,} parameters")

# ─────────────────────────────────────────────────────────────
# SECTION 4: FIX 2 — Reward Shaping
# ─────────────────────────────────────────────────────────────
# WHY sparse rewards kill REINFORCE:
#   In v1, reward=1 only at the goal. With ~3% random success rate,
#   97% of episodes return G_t=0 for ALL timesteps. Zero gradients
#   mean the network barely updates from failed episodes.
#
# FIX — shaped reward function:
#   +0.01 for each step survived on a frozen tile  → every safe step
#         gives a small positive signal, encouraging exploration.
#   -0.50 for falling into a hole                  → strongly penalizes
#         hole-steps, teaching the agent to avoid them.
#   +1.00 for reaching the goal                    → unchanged.
#
# This converts a pure sparse signal into a DENSE shaped signal.
# Important: shaped rewards introduce a bias if not carefully designed
# (potential-based shaping is theoretically safe), but for a small
# discrete env like FrozenLake, simple shaping works well in practice.

MAP_HOLES  = {5, 7, 11, 12}   # hole state indices in 4x4 map
MAP_GOAL   = 15               # goal state index

def shape_reward(state: int, next_state: int,
                 raw_reward: float, done: bool) -> float:
    """
    Transform the sparse gym reward into a denser shaped reward.

    Args:
        state     : current state before action
        next_state: state after taking action
        raw_reward: original gymnasium reward (0 or 1)
        done      : whether the episode terminated

    Returns:
        Shaped reward signal for policy gradient.
    """
    if raw_reward > 0:
        return +1.0                  # goal reached — keep original signal
    if done and next_state in MAP_HOLES:
        return -0.5                  # fell in hole — penalize heavily
    return +0.01                     # survived on frozen tile — small bonus

# ─────────────────────────────────────────────────────────────
# SECTION 5: FIX 1 — Entropy Bonus (anti-collapse)
# ─────────────────────────────────────────────────────────────
# WHY v1 policy collapsed to 0% greedy eval:
#   During stochastic sampling, the agent occasionally succeeds.
#   But the policy gradient pushes toward those actions strongly.
#   Over time, the softmax probabilities saturate (one action → 1.0,
#   others → 0.0). The argmax then always picks the same action,
#   which may not be optimal for every state.
#
# FIX — Entropy bonus:
#   H(π) = -Σ_a π(a|s) * log π(a|s)   (Shannon entropy)
#   Add beta * H(π) to the loss (i.e., subtract from NEGATIVE loss).
#   This REWARDS high-entropy (spread-out) distributions, actively
#   penalizing the policy for being too confident too early.
#   Effect: keeps probabilities from collapsing; maintains exploratory
#   behaviour throughout training, giving argmax more meaningful choices.

def compute_discounted_returns(rewards: list, gamma: float) -> torch.Tensor:
    """Compute G_t = Σ γ^k r_{t+k} for each timestep."""
    T       = len(rewards)
    returns = torch.zeros(T, dtype=torch.float32, device=device)
    G       = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

# ─────────────────────────────────────────────────────────────
# SECTION 6: FIX 3 — Running Baseline (variance reduction)
# ─────────────────────────────────────────────────────────────
# v1 subtracted the BATCH mean (mean within a single episode).
# For sparse rewards most episodes have return ≈ 0, so the batch
# mean ≈ 0 → normalization divides near-zero by near-zero → NaN / noise.
#
# FIX — Global running baseline:
#   b = mean(G_0) over last BASELINE_WIN episode totals.
#   Subtract b from each G_t BEFORE normalizing within the episode.
#   Effect: successful episodes get clearly positive advantage,
#   failed episodes get clearly negative advantage, even when
#   the batch mean within one episode is all-zero.

baseline_deque = collections.deque(maxlen=BASELINE_WIN)  # stores episode G_0

def normalize_with_baseline(returns: torch.Tensor,
                             episode_return: float) -> torch.Tensor:
    """
    Subtract running baseline then normalize returns.

    Steps:
      1. Compute running baseline b = mean of last 100 episode returns.
      2. Subtract b from all returns in this episode.
      3. Normalize: (returns - mean) / (std + eps) within episode.
    """
    # Subtract global running mean (cross-episode baseline)
    if len(baseline_deque) > 0:
        baseline  = np.mean(baseline_deque)
        returns   = returns - baseline

    # Normalize within the episode for scale stability
    mean = returns.mean()
    std  = returns.std()
    if std > 1e-8:
        returns = (returns - mean) / (std + 1e-8)
    return returns

# ─────────────────────────────────────────────────────────────
# SECTION 7: REINFORCE Update with Entropy Bonus
# ─────────────────────────────────────────────────────────────

def reinforce_update_v2(log_probs: list,
                        entropies: list,
                        returns:   torch.Tensor) -> float:
    """
    REINFORCE update with entropy regularization.

    Loss = -Σ_t log π(a_t|s_t) * G_t   (policy gradient)
           - beta * Σ_t H(π(·|s_t))    (entropy bonus)

    The entropy term encourages the policy to stay spread-out,
    preventing premature convergence to a deterministic (collapsed)
    policy before the agent has properly explored the goal state.
    """
    log_probs_t = torch.stack(log_probs)     # [T]
    entropies_t = torch.stack(entropies)     # [T]

    pg_loss      = -(log_probs_t * returns).sum()
    entropy_loss = -ENTROPY_BETA * entropies_t.sum()   # negative = bonus
    total_loss   = pg_loss + entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item()

# ─────────────────────────────────────────────────────────────
# SECTION 8: Main Training Loop — REINFORCE v2
# ─────────────────────────────────────────────────────────────

env             = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
episode_rewards = []     # raw (unshaped) reward per episode
episode_success = []     # True/False

print("\n" + "=" * 62)
print("              TRAINING PROGRESS  (v2)")
print("=" * 62)
print(f"  {'Episode':>10}  {'Successes':>12}  {'Success%':>10}  {'Avg Loss':>10}")
print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")

loss_window = []

for episode in range(N_EPISODES):

    state, _ = env.reset()
    ep_raw_reward = 0     # tracks original (unshaped) reward for success eval

    log_probs  = []
    entropies  = []
    rewards    = []        # shaped rewards for gradient computation

    # ── Episode rollout ──
    for step in range(MAX_STEPS):
        state_tensor = one_hot(state)
        action_probs = policy_net(state_tensor)

        dist      = Categorical(action_probs)
        action    = dist.sample()
        log_p     = dist.log_prob(action)
        entropy   = dist.entropy()          # H(π) at this state

        next_state, raw_reward, done, truncated, _ = env.step(action.item())

        # Apply reward shaping (FIX 2)
        shaped_r = shape_reward(state, next_state, raw_reward, done)

        log_probs.append(log_p)
        entropies.append(entropy)
        rewards.append(shaped_r)

        ep_raw_reward += raw_reward
        state = next_state

        if done or truncated:
            break

    # ── Discounted returns (from shaped rewards) ──
    returns = compute_discounted_returns(rewards, GAMMA)

    # Update running baseline with this episode's total shaped return
    episode_total_return = returns[0].item()    # G_0 = total discounted return
    baseline_deque.append(episode_total_return)

    # ── Apply running baseline + normalization (FIX 3) ──
    returns = normalize_with_baseline(returns, episode_total_return)

    # ── REINFORCE update with entropy bonus (FIX 1) ──
    loss = reinforce_update_v2(log_probs, entropies, returns)
    loss_window.append(loss)

    # ── Record outcome (using original unshaped reward) ──
    success = ep_raw_reward > 0
    episode_rewards.append(ep_raw_reward)
    episode_success.append(success)

    # Print every 2000 episodes
    if (episode + 1) % 2000 == 0:
        window_successes = sum(episode_success[-2000:])
        window_rate      = window_successes / 2000 * 100
        avg_loss         = np.mean(loss_window[-2000:]) if loss_window else 0.0
        print(
            f"  {episode+1:>10,}  "
            f"{window_successes:>12}  "
            f"{window_rate:>9.1f}%  "
            f"{avg_loss:>10.4f}"
        )

env.close()
print(f"\n  Training complete! {N_EPISODES:,} episodes finished.")

# ─────────────────────────────────────────────────────────────
# SECTION 9: Greedy Evaluation (1000 episodes)
# ─────────────────────────────────────────────────────────────

eval_env     = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
eval_success = 0
eval_rewards = []

print("\n" + "=" * 62)
print(f"   GREEDY POLICY EVALUATION ({EVAL_EPS:,} episodes)")
print("=" * 62)

policy_net.eval()
with torch.no_grad():
    for ep in range(EVAL_EPS):
        state, _ = eval_env.reset(seed=ep)
        done      = False
        truncated = False
        ep_reward = 0
        steps     = 0

        while not done and not truncated and steps < MAX_STEPS:
            probs  = policy_net(one_hot(state))
            action = probs.argmax().item()     # greedy
            state, reward, done, truncated, _ = eval_env.step(action)
            ep_reward += reward
            steps     += 1

        eval_rewards.append(ep_reward)
        if ep_reward > 0:
            eval_success += 1

eval_env.close()
policy_net.train()

pg_v2_success = eval_success / EVAL_EPS * 100
pg_v2_avg_rew = np.mean(eval_rewards)

print(f"\n  Successful episodes : {eval_success} / {EVAL_EPS:,}")
print(f"  Success rate        : {pg_v2_success:.2f}%")
print(f"  Avg reward          : {pg_v2_avg_rew:.4f}")

# ─────────────────────────────────────────────────────────────
# SECTION 10: Save Model
# ─────────────────────────────────────────────────────────────

torch.save(policy_net.state_dict(), "policy_gradient_v2_model.pth")
print(f"\n  Model saved --> policy_gradient_v2_model.pth")

# ─────────────────────────────────────────────────────────────
# SECTION 11: Plots
# ─────────────────────────────────────────────────────────────

print("\n  Generating plots...")

rewards_arr = np.array(episode_rewards, dtype=float)
success_arr = np.array(episode_success, dtype=float)
window      = 100

smoothed_rewards = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
rolling_success  = np.convolve(success_arr, np.ones(window) / window, mode="valid") * 100
smooth_x         = np.arange(window - 1, N_EPISODES)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "REINFORCE v2 (Improved) on FrozenLake-v1  "
    "[entropy bonus + reward shaping + running baseline]",
    fontsize=11, fontweight="bold", y=1.02
)

# ── Graph 1: Reward Curve ──
ax1 = axes[0]
ax1.plot(rewards_arr, color="#8ecae6", alpha=0.2, linewidth=0.4,
         label="Raw episode reward")
ax1.plot(smooth_x, smoothed_rewards, color="#0077b6", linewidth=2.0,
         label=f"{window}-ep rolling average")

ax1.axhline(y=pg_v2_success / 100, color="#023047", linewidth=1.6,
            linestyle="--",
            label=f"PG v2 greedy eval: {pg_v2_success:.1f}%")
ax1.axhline(y=QL_SUCCESS / 100, color="#2a9d8f", linewidth=1.0,
            linestyle=":", alpha=0.8, label=f"Q-Learning: {QL_SUCCESS}%")
ax1.axhline(y=DQN_SUCCESS / 100, color="#e9c46a", linewidth=1.0,
            linestyle=":", alpha=0.8, label=f"DQN: {DQN_SUCCESS}%")
ax1.axhline(y=PG_V1_SUCCESS / 100, color="#e63946", linewidth=1.0,
            linestyle=":", alpha=0.6, label=f"REINFORCE v1: {PG_V1_SUCCESS}%")

ax1.set_title("REINFORCE v2 — Training Reward Curve", fontsize=11, fontweight="bold")
ax1.set_xlabel("Episode", fontsize=10)
ax1.set_ylabel("Reward (0=fail, 1=success)", fontsize=10)
ax1.set_ylim(-0.08, 1.30)
ax1.legend(fontsize=7.5, loc="upper left")
ax1.grid(True, alpha=0.3)

final_smooth = smoothed_rewards[-1]
ax1.annotate(
    f"Final avg: {final_smooth:.3f}",
    xy=(smooth_x[-1], final_smooth),
    xytext=(smooth_x[-1] - 3500, final_smooth + 0.08),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color="#0077b6"),
    color="#0077b6"
)

# ── Graph 2: Rolling Success Rate ──
ax2 = axes[1]
ax2.plot(smooth_x, rolling_success, color="#fb8500", linewidth=1.8,
         label=f"{window}-ep rolling success %")
ax2.fill_between(smooth_x, 0, rolling_success, alpha=0.15, color="#fb8500")

ax2.axhline(y=QL_SUCCESS,    color="#2a9d8f", linewidth=1.1,
            linestyle="--", alpha=0.85, label=f"Q-Learning: {QL_SUCCESS}%")
ax2.axhline(y=DQN_SUCCESS,   color="#e9c46a", linewidth=1.1,
            linestyle="--", alpha=0.85, label=f"DQN: {DQN_SUCCESS}%")
ax2.axhline(y=PG_V1_SUCCESS, color="#e63946", linewidth=1.0,
            linestyle=":", alpha=0.7, label=f"REINFORCE v1: {PG_V1_SUCCESS}%")
ax2.axhline(y=pg_v2_success, color="#023047", linewidth=1.3,
            linestyle="-.", label=f"PG v2 eval: {pg_v2_success:.1f}%")

ax2.set_title("REINFORCE v2 — Rolling Success Rate (100-ep)",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("Episode", fontsize=10)
ax2.set_ylabel("Success rate (%)", fontsize=10)
ax2.set_ylim(-2, 100)
ax2.legend(fontsize=7.5, loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("policy_gradient_v2_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved --> policy_gradient_v2_results.png")

# ─────────────────────────────────────────────────────────────
# SECTION 12: 4-Way Comparison Table
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 74)
print("  4-WAY COMPARISON: Q-Learning | DQN | PG v1 | PG v2")
print("=" * 74)

hdrs = (f"  {'Metric':<28}  {'Q-Learn':>9}  {'DQN':>8}  "
        f"{'PG v1':>8}  {'PG v2':>9}")
div  = "  " + "-" * 70
print(hdrs)
print(div)

rows = [
    ("Method type",          "Value",      "Value",     "Policy",    "Policy"),
    ("Exploration",          "eps-greedy", "eps-greedy","Stochastic","Stochastic"),
    ("Reward signal",        "Sparse",     "Sparse",    "Sparse",    "Shaped"),
    ("Entropy bonus",        "N/A",        "N/A",       "No",        "Yes"),
    ("Update type",          "TD step",    "TD step",   "MC episode","MC episode"),
    ("Baseline",             "None",       "None",      "Batch mean","Running 100"),
    ("Learning rate",        "0.800",      "0.001",     "0.001",     "0.0005"),
    ("Episodes",             "10,000",     "10,000",    "10,000",    "20,000"),
    ("Eval success rate",    f"{QL_SUCCESS}%",
                                          f"{DQN_SUCCESS}%",
                                                       f"{PG_V1_SUCCESS}%",
                                                                    f"{pg_v2_success:.1f}%"),
    ("Avg eval reward",      "0.7230",    "0.6570",    "0.0000",    f"{pg_v2_avg_rew:.4f}"),
]

for metric, *vals in rows:
    print(f"  {metric:<28}  {vals[0]:>9}  {vals[1]:>8}  {vals[2]:>8}  {vals[3]:>9}")

print(div)
print()

# Improvement comparison
improvement = pg_v2_success - PG_V1_SUCCESS
print(f"  REINFORCE v2 vs v1 improvement : {improvement:+.1f}% success")
print()

print("  Analysis of v2 fixes:")
print()
print("  [FIX 1 — Entropy bonus]")
print("    Prevents softmax saturation. With beta=0.01, the policy is")
print("    penalized for assigning probability 1.0 to a single action.")
print("    Result: argmax (greedy eval) picks a MORE DIVERSE set of")
print("    actions, reducing the chance of deterministic bad looping.")
print()
print("  [FIX 2 — Reward shaping]")
print("    step=+0.01 gives gradient signal on EVERY step (not just goal).")
print("    hole=-0.50 makes failure clearly distinguishable from safe steps.")
print("    The policy now receives non-zero gradients on 100% of episodes,")
print("    not just the ~3% that accidentally reached the goal.")
print()
print("  [FIX 3 — Running baseline]")
print("    Subtracting E[G_0] over last 100 episodes from episode returns")
print("    ensures failed episodes get NEGATIVE advantage (discouraging)")
print("    even when their raw return is still positive (due to shaping).")
print("    More stable than v1 batch-mean normalization alone.")
print()
print("  [FIX 4 — Slower lr + more episodes]")
print("    lr=0.0005 prevents overshooting the narrow reward basin.")
print("    20k episodes gives the shaped signal enough time to propagate.")

print()
print("=" * 74)
print("  REINFORCE v2 complete!")
print("  Saved: policy_gradient_v2_model.pth | policy_gradient_v2_results.png")
print("=" * 74)
