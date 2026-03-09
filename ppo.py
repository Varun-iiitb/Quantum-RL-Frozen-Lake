"""
=============================================================
  Phase 2: PPO (Proximal Policy Optimization) — FrozenLake-v1
           Using Stable-Baselines3
=============================================================
  Quantum RL Project — FrozenLake Edition
  Author : Varun E
  Date   : March 2026

  Algorithm: PPO  [Schulman et al., 2017]
  Library  : stable-baselines3

  WHY PPO IS BETTER THAN VANILLA REINFORCE:
  ─────────────────────────────────────────
  REINFORCE updates the policy in whatever direction the gradient
  points, with NO constraint on how large the update is.
  On sparse reward environments this is catastrophic:
    - A single lucky episode pushes the policy too aggressively.
    - One large update can destroy previously learned behaviour.
    - The policy oscillates without converging.

  PPO fixes this with the CLIPPED SURROGATE OBJECTIVE:

    L_CLIP(θ) = E_t [ min(r_t(θ)*A_t,  clip(r_t(θ), 1-ε, 1+ε)*A_t) ]

  where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
        A_t    = advantage estimate (how much better than baseline)
        ε      = 0.2  (clip range)

  The clip(·) prevents r_t from going too far from 1.0, which means:
    - Policy updates are bounded regardless of gradient magnitude.
    - The policy cannot "overfit" to a single lucky trajectory.
    - Training is stable across many gradient steps on the same batch.

  Additional PPO advantages over REINFORCE:
    - GAE (Generalized Advantage Estimation): lower-variance advantage
      estimates using a learned value function (actor-critic).
    - Multiple epochs per rollout: reuses experience safely (clipping
      prevents overfitting), making it far more sample efficient.
    - Entropy bonus: built-in, maintains exploration automatically.

  Combined effect on FrozenLake (sparse + stochastic):
    - Value function (critic) learns good baselines quickly, reducing
      the variance that crippled REINFORCE.
    - Conservative updates prevent the collapse seen in PG v1/v2.
    - Multiple epochs per rollout squeeze more signal from rare
      successful episodes.

  Hyperparameters:
    policy       = MlpPolicy
    learning_rate= 0.0003
    n_steps      = 2048
    batch_size   = 64
    n_epochs     = 10
    gamma        = 0.99
    timesteps    = 500,000
=============================================================
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Constants & Hyperparameters
# ─────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 500_000
EVAL_EPISODES   = 1_000
LOG_INTERVAL    = 1_000    # record stats every N timesteps

# Previous algorithm results (for comparison)
QL_SUCCESS    = 72.3
DQN_SUCCESS   = 65.7
PG_V1_SUCCESS =  0.0
PG_V2_SUCCESS =  0.0

print("=" * 62)
print("    PHASE 2: PPO AGENT ON FROZENLAKE-v1  (SB3)")
print("=" * 62)
print(f"\n  Algorithm : PPO  (Stable-Baselines3)")
print(f"  Policy    : MlpPolicy  (actor + critic MLPs)")
print(f"\n  PPO Hyperparameters:")
print(f"    learning_rate = 0.0003")
print(f"    n_steps       = 2048   (rollout buffer size)")
print(f"    batch_size    = 64     (mini-batch for gradient updates)")
print(f"    n_epochs      = 10     (epochs per rollout)")
print(f"    gamma         = 0.99")
print(f"    clip_range    = 0.2    (PPO epsilon — clips policy ratio)")
print(f"    total_steps   = {TOTAL_TIMESTEPS:,}")
print(f"    eval_episodes = {EVAL_EPISODES:,}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Environment Setup
# ─────────────────────────────────────────────────────────────
# Monitor wrapper adds episode reward/length tracking that
# Stable-Baselines3 uses internally for logging.

env = Monitor(
    gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
)
print(f"\n  Environment : FrozenLake-v1  (4x4, is_slippery=True)")
print(f"  Wrapped in  : SB3 Monitor (episode logging)")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Custom Training Callback
# ─────────────────────────────────────────────────────────────
# Stable-Baselines3 uses callbacks to hook into the training loop.
# We record episode reward and success every LOG_INTERVAL timesteps
# to build our own reward/success curves for plotting.

class EpisodeTrackingCallback(BaseCallback):
    """
    Custom SB3 callback that collects per-episode metrics during training.

    SB3 calls on_step() after every environment step. We accumulate
    completed episode data from the Monitor wrapper's info dict.

    Stored data:
        timestep_log  : timestep at which each episode completed
        reward_log    : total reward of that episode (0 or 1)
        success_log   : True/False for that episode
    """

    def __init__(self, log_interval: int = LOG_INTERVAL, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval  = log_interval
        self.timestep_log  = []
        self.reward_log    = []
        self.success_log   = []
        self._last_print   = 0
        self._success_window = []   # recent episodes for progress printing

    def _on_step(self) -> bool:
        """Called after every env.step() during training."""
        # SB3 Monitor stores episode info in self.locals["infos"]
        # "episode" key is added by Monitor when an episode ends
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_success = bool(ep_reward > 0)

                self.timestep_log.append(self.num_timesteps)
                self.reward_log.append(ep_reward)
                self.success_log.append(ep_success)
                self._success_window.append(ep_success)

        # Print training progress every log_interval timesteps
        if self.num_timesteps - self._last_print >= self.log_interval * 50:
            # Compute recent success rate from last 500 completed episodes
            recent = self._success_window[-500:] if self._success_window else [False]
            rate   = sum(recent) / len(recent) * 100
            ep_cnt = len(self.timestep_log)
            print(
                f"  Step {self.num_timesteps:>9,}  |  "
                f"Episodes: {ep_cnt:>6,}  |  "
                f"Recent success: {rate:>5.1f}%"
            )
            self._last_print = self.num_timesteps

        return True   # returning False stops training early


# ─────────────────────────────────────────────────────────────
# SECTION 4: Build PPO Model
# ─────────────────────────────────────────────────────────────
# SB3 PPO uses an actor-critic architecture by default:
#   - Actor (policy net) : decides which action to take
#   - Critic (value net) : estimates V(s) to compute advantages
# Both are MLPs with 2 hidden layers of 64 neurons by default.

model = PPO(
    policy        = "MlpPolicy",
    env           = env,
    learning_rate = 0.0003,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    clip_range    = 0.2,       # PPO clipping epsilon
    ent_coef      = 0.01,      # entropy bonus coefficient (like REINFORCE v2)
    verbose       = 0,         # suppress SB3 default logs (we use callback)
    device        = "cpu",
)

n_params = sum(p.numel() for p in model.policy.parameters())
print(f"\n  PPO actor-critic network parameters: {n_params:,}")
print(f"  (actor MLP + critic MLP sharing feature extractor)")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Train PPO
# ─────────────────────────────────────────────────────────────

callback = EpisodeTrackingCallback(log_interval=LOG_INTERVAL)

print("\n" + "=" * 62)
print("              TRAINING PROGRESS  (PPO)")
print("=" * 62)
print(f"  {'Timestep':>12}  |  {'Episodes':>8}  |  {'Recent %':>10}")
print(f"  {'-'*12}  |  {'-'*8}  |  {'-'*10}")

model.learn(
    total_timesteps  = TOTAL_TIMESTEPS,
    callback         = callback,
    progress_bar     = False,
)

print(f"\n  Training complete! {TOTAL_TIMESTEPS:,} timesteps finished.")
print(f"  Total episodes seen: {len(callback.reward_log):,}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Evaluate Trained PPO Policy
# ─────────────────────────────────────────────────────────────
# SB3's evaluate_policy runs deterministic (greedy) episodes and
# returns mean_reward and std_reward over n_eval_episodes.

eval_env = Monitor(
    gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
)

print("\n" + "=" * 62)
print(f"   GREEDY POLICY EVALUATION ({EVAL_EPISODES:,} episodes)")
print("=" * 62)

# Collect individual episode rewards for success rate
eval_rewards = []
eval_success = 0

obs, _ = eval_env.reset()
done = truncated = False
ep_reward = 0

for _ in range(EVAL_EPISODES):
    obs, _ = eval_env.reset()
    done = truncated = False
    ep_reward = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)  # greedy
        obs, reward, done, truncated, _ = eval_env.step(int(action))
        ep_reward += reward
    eval_rewards.append(ep_reward)
    if ep_reward > 0:
        eval_success += 1

eval_env.close()

ppo_success_rate = eval_success / EVAL_EPISODES * 100
ppo_avg_reward   = np.mean(eval_rewards)

print(f"\n  Successful episodes : {eval_success} / {EVAL_EPISODES:,}")
print(f"  Success rate        : {ppo_success_rate:.2f}%")
print(f"  Avg reward          : {ppo_avg_reward:.4f}")

# ─────────────────────────────────────────────────────────────
# SECTION 7: Save Model
# ─────────────────────────────────────────────────────────────

model.save("ppo_model")
print(f"\n  Model saved --> ppo_model.zip")

# ─────────────────────────────────────────────────────────────
# SECTION 8: Plots
# ─────────────────────────────────────────────────────────────

print("\n  Generating plots...")

timesteps    = np.array(callback.timestep_log)
raw_rewards  = np.array(callback.reward_log, dtype=float)
raw_success  = np.array(callback.success_log, dtype=float)

# Guard: need at least `window` episodes to smooth
window = 200
if len(raw_rewards) >= window:
    smoothed_rewards = np.convolve(raw_rewards, np.ones(window) / window, mode="valid")
    rolling_success  = np.convolve(raw_success, np.ones(window) / window, mode="valid") * 100
    ts_smooth        = timesteps[window - 1:]
else:
    smoothed_rewards = raw_rewards
    rolling_success  = raw_success * 100
    ts_smooth        = timesteps

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"PPO on FrozenLake-v1  (SB3, MlpPolicy, {TOTAL_TIMESTEPS//1000}k timesteps)",
    fontsize=12, fontweight="bold", y=1.02
)

# ── Graph 1: Episode Reward Smoothed Curve ──
ax1 = axes[0]
ax1.scatter(timesteps, raw_rewards, color="#adb5bd", s=1, alpha=0.15,
            label="Raw episode reward")
ax1.plot(ts_smooth, smoothed_rewards, color="#4361ee", linewidth=2.0,
         label=f"{window}-ep rolling average")
ax1.axhline(y=ppo_success_rate / 100, color="#023047", linewidth=1.6,
            linestyle="--",
            label=f"PPO greedy eval: {ppo_success_rate:.1f}%")
ax1.axhline(y=QL_SUCCESS / 100,    color="#2a9d8f", linewidth=1.0,
            linestyle=":", alpha=0.8, label=f"Q-Learning: {QL_SUCCESS}%")
ax1.axhline(y=DQN_SUCCESS / 100,   color="#e9c46a", linewidth=1.0,
            linestyle=":", alpha=0.8, label=f"DQN: {DQN_SUCCESS}%")

ax1.set_title("PPO — Training Reward Curve", fontsize=11, fontweight="bold")
ax1.set_xlabel("Training Timesteps", fontsize=10)
ax1.set_ylabel("Episode Reward (0=fail, 1=success)", fontsize=10)
ax1.set_ylim(-0.05, 1.30)
ax1.legend(fontsize=7.5, loc="upper left")
ax1.grid(True, alpha=0.3)

# Annotate peak average
if len(smoothed_rewards) > 0:
    peak_idx = np.argmax(smoothed_rewards)
    ax1.annotate(
        f"Peak avg: {smoothed_rewards[peak_idx]:.3f}",
        xy=(ts_smooth[peak_idx], smoothed_rewards[peak_idx]),
        xytext=(ts_smooth[peak_idx] + 10000,
                smoothed_rewards[peak_idx] + 0.06),
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color="#4361ee"),
        color="#4361ee"
    )

# ── Graph 2: Rolling Success Rate ──
ax2 = axes[1]
ax2.plot(ts_smooth, rolling_success, color="#f72585", linewidth=1.8,
         label=f"{window}-ep rolling success %")
ax2.fill_between(ts_smooth, 0, rolling_success, alpha=0.12, color="#f72585")

ax2.axhline(y=QL_SUCCESS,       color="#2a9d8f", linewidth=1.1,
            linestyle="--", alpha=0.85, label=f"Q-Learning: {QL_SUCCESS}%")
ax2.axhline(y=DQN_SUCCESS,      color="#e9c46a", linewidth=1.1,
            linestyle="--", alpha=0.85, label=f"DQN: {DQN_SUCCESS}%")
ax2.axhline(y=ppo_success_rate, color="#023047", linewidth=1.4,
            linestyle="-.", label=f"PPO eval: {ppo_success_rate:.1f}%")

ax2.set_title("PPO — Rolling Success Rate (200-ep window)",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("Training Timesteps", fontsize=10)
ax2.set_ylabel("Success rate (%)", fontsize=10)
ax2.set_ylim(-2, 100)
ax2.legend(fontsize=7.5, loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ppo_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved --> ppo_results.png")

# ─────────────────────────────────────────────────────────────
# SECTION 9: Full 5-Way Comparison Table
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 76)
print("   FULL COMPARISON: Q-Learning | DQN | PG v1 | PG v2 | PPO")
print("=" * 76)

hdrs = (f"  {'Metric':<30}  {'Q-Learn':>8}  {'DQN':>7}  "
        f"{'PG v1':>7}  {'PG v2':>7}  {'PPO':>8}")
div  = "  " + "-" * 73
print(hdrs)
print(div)

rows = [
    ("Method type",        "Value",    "Value",    "Policy",   "Policy",   "Actor-Crit"),
    ("Library",            "Custom",   "PyTorch",  "PyTorch",  "PyTorch",  "SB3"),
    ("Advantage estimate", "TD",       "TD",       "MC",       "MC+base",  "GAE"),
    ("Policy update",      "Greedy",   "Greedy",   "Gradient", "Gradient", "Clipped PG"),
    ("Update safety",      "None",     "T-Net",    "None",     "None",     "ε-clip=0.2"),
    ("Entropy bonus",      "No",       "No",       "No",       "Yes",      "Yes"),
    ("Reward signal",      "Sparse",   "Sparse",   "Sparse",   "Shaped",   "Sparse"),
    ("Episodes/timesteps", "10k ep",   "10k ep",   "10k ep",   "20k ep",   "500k ts"),
    ("Eval success rate",  f"{QL_SUCCESS}%",
                                       f"{DQN_SUCCESS}%",
                                                   f"{PG_V1_SUCCESS}%",
                                                               f"{PG_V2_SUCCESS}%",
                                                                           f"{ppo_success_rate:.1f}%"),
    ("Avg eval reward",    "0.723",    "0.657",    "0.000",    "0.000",    f"{ppo_avg_reward:.3f}"),
]

for metric, *vals in rows:
    print(f"  {metric:<30}  {vals[0]:>8}  {vals[1]:>7}  "
          f"{vals[2]:>7}  {vals[3]:>7}  {vals[4]:>8}")

print(div)

# Ranking
scores = {
    "Q-Learning": QL_SUCCESS,
    "DQN":        DQN_SUCCESS,
    "REINFORCE":  PG_V1_SUCCESS,
    "REINFORCE-v2": PG_V2_SUCCESS,
    "PPO":        ppo_success_rate,
}
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print(f"\n  RANKING (by greedy eval success rate):")
for i, (name, score) in enumerate(ranked, 1):
    bar  = "#" * int(score / 2)
    note = " <-- best" if i == 1 else ""
    print(f"    {i}. {name:<14} {score:>6.1f}%  {bar}{note}")

print()
print("=" * 76)
print()
print("  KEY INSIGHTS:")
print()
print("  WHY PPO > REINFORCE on FrozenLake:")
print("    1. GAE advantage estimates use the VALUE FUNCTION as a baseline.")
print("       This dramatically reduces variance vs. Monte-Carlo G_t.")
print("       Sparse rewards are less catastrophic when variance is managed.")
print()
print("    2. CLIPPED SURROGATE OBJECTIVE bounds each update step.")
print("       r_t = π_new(a|s)/π_old(a|s) is clipped to [0.8, 1.2].")
print("       Even when a lucky episode produces a large gradient,")
print("       the clip prevents the policy from changing too drastically.")
print("       => Stable convergence that REINFORCE w/ large gradients lacks.")
print()
print("    3. MULTIPLE EPOCHS per rollout reuses each trajectory 10 times.")
print("       On sparse-reward envs where successes are rare,")
print("       squeezing 10 gradient updates from one success is critical.")
print("       REINFORCE discards each trajectory after 1 update (wasteful).")
print()
print("    4. ENTROPY BONUS (ent_coef=0.01) prevents action collapse,")
print("       the same fix we manually added to REINFORCE v2.")
print()
print("  WHY Q-LEARNING STILL TOPS THE CHART:")
print("    On a tiny 16-state environment, the Q-table stores EXACT")
print("    Q-values with no approximation error. No neural network,")
print("    no function approximator — pure lookup. Hard to beat.")
print("    PPO (and all neural methods) are designed for problems where")
print("    the state space is too large for a table.")
print()
print("=" * 76)
print(f"  PPO complete!  Saved: ppo_model.zip | ppo_results.png")
print("=" * 76)

env.close()
