"""
=============================================================
  Phase 1: FrozenLake-v1 Environment Setup & Exploration
=============================================================
  Quantum RL Project -- FrozenLake Edition
  Author: Varun E
  Date  : March 2026

  This script sets up the FrozenLake-v1 Gymnasium environment,
  explores its structure, runs 5 random episodes, visualizes
  the grid, and explains why FrozenLake is challenging for RL.
=============================================================
"""

import sys
import gymnasium as gym
import numpy as np

# Force UTF-8 output so emojis/special chars print on Windows
sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Initialize the FrozenLake-v1 Environment
# ─────────────────────────────────────────────────────────────
# map_name="4x4"    -> use the standard 4x4 grid
# is_slippery=True  -> stochastic transitions (the agent can
#                     slip and move in unintended directions)
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
    render_mode="ansi"   # text-based rendering
)

print("=" * 62)
print("         FROZENLAKE-v1 ENVIRONMENT OVERVIEW")
print("=" * 62)

# ─────────────────────────────────────────────────────────────
# SECTION 2: Display Environment Metadata
# ─────────────────────────────────────────────────────────────
num_states  = env.observation_space.n   # total number of discrete states
num_actions = env.action_space.n        # total number of discrete actions

print(f"\n  Observation Space : {env.observation_space}  -> {num_states} discrete states")
print(f"  Action Space      : {env.action_space}  -> {num_actions} actions")
print(f"  Actions           : 0=LEFT   1=DOWN   2=RIGHT   3=UP")
print(f"\n  Grid Size         : 4x4  (map_name='4x4')")
print(f"  Is Slippery       : True  (stochastic environment)")
print(f"  Note: Each intended move succeeds with only 1/3 probability.")
print(f"        The agent may slip 90 degrees left/right with 1/3 each.")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Render the 4x4 Grid
# ─────────────────────────────────────────────────────────────
# Legend:
#   S = Start (top-left, safe)
#   F = Frozen (safe to walk on)
#   H = Hole   (agent falls -> episode ends, reward = 0)
#   G = Goal   (agent wins  -> episode ends, reward = 1)

print("\n" + "=" * 62)
print("              4x4 FROZENLAKE GRID MAP")
print("=" * 62)

# Standard 4x4 FrozenLake map layout
MAP_4x4 = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

# Visual representation using bordered cells
cell_display = {
    "S": "[ S ]",  # Start
    "F": "[ F ]",  # Frozen
    "H": "[ H ]",  # Hole
    "G": "[ G ]",  # Goal
}

print()
print("       Col0   Col1   Col2   Col3")
print("     +------+------+------+------+")
for row_idx, row in enumerate(MAP_4x4):
    cells = " | ".join(cell_display[cell] for cell in row)
    print(f"Row {row_idx} | {cells} |")
    print("     +------+------+------+------+")

print()
print("  Legend:")
print("    [ S ] = Start position  (top-left, state 0)")
print("    [ F ] = Frozen tile     (safe to walk on)")
print("    [ H ] = Hole            (agent falls, reward=0, episode ends)")
print("    [ G ] = Goal            (reward=1, episode ends)")
print()
print("  State numbering: row * 4 + col  (0 to 15)")
print("  e.g. state 5 = row 1, col 1  (the first Hole)")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Run 5 Random Episodes
# ─────────────────────────────────────────────────────────────
# A random policy samples actions uniformly -- naive baseline
# before any training. This shows how hard the environment is.

NUM_EPISODES = 5
episode_rewards = []  # cumulative reward for each episode
episode_success = []  # whether goal was reached each episode

print("\n" + "=" * 62)
print("        RANDOM AGENT -- 5 EXPLORATION EPISODES")
print("=" * 62)
print("  Policy: random action sampled each step (no learning)")

action_names = {0: "LEFT ", 1: "DOWN ", 2: "RIGHT", 3: "UP   "}

for episode in range(NUM_EPISODES):
    # Reset to start a fresh episode; seed ensures reproducibility
    state, info = env.reset(seed=42 + episode)

    total_reward = 0   # cumulative reward this episode
    step_count   = 0   # number of steps taken
    done         = False
    truncated    = False

    print(f"\n  {'─'*58}")
    print(f"  EPISODE {episode + 1}  |  Starting state: {state} "
          f"(row={state//4}, col={state%4})")
    print(f"  {'─'*58}")
    print(f"  {'Step':>4}  {'From':>8}  {'To':>8}  {'Action':>7}  {'Reward':>7}  Outcome")
    print(f"  {'----':>4}  {'--------':>8}  {'------':>8}  {'------':>7}  {'------':>7}  -------")

    while not done and not truncated:
        # Sample a random action from the action space
        action = env.action_space.sample()

        # Take the action in the environment
        # Returns: next_state, reward, terminated, truncated, info
        next_state, reward, done, truncated, info = env.step(action)

        step_count   += 1
        total_reward += reward

        # Build a human-readable outcome for this step
        if done and reward > 0:
            outcome = "*** GOAL REACHED! ***"
        elif done:
            outcome = "! Fell in hole"
        elif truncated:
            outcome = "! Time limit hit"
        else:
            outcome = ""

        from_str = f"({state//4},{state%4})[{state:>2}]"
        to_str   = f"({next_state//4},{next_state%4})[{next_state:>2}]"

        print(
            f"  {step_count:>4}  "
            f"{from_str:>8}  "
            f"{to_str:>8}  "
            f"{action_names[action]:>7}  "
            f"{reward:>7.1f}  "
            f"{outcome}"
        )

        state = next_state   # advance to next state

    # ── Episode-level summary ──
    success = done and total_reward > 0
    episode_rewards.append(total_reward)
    episode_success.append(success)

    status_str = "SUCCESS -- Goal reached!" if success else "FAILED  -- Did not reach goal"
    print(f"\n  >> Steps taken  : {step_count}")
    print(f"  >> Total reward : {total_reward:.1f}")
    print(f"  >> Outcome      : {status_str}")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Overall Summary Across All Episodes
# ─────────────────────────────────────────────────────────────
total_reward_all = sum(episode_rewards)
num_successes    = sum(episode_success)
success_rate     = num_successes / NUM_EPISODES * 100

print("\n" + "=" * 62)
print("           SUMMARY -- 5 RANDOM EPISODES")
print("=" * 62)
print(f"\n  Policy               : Fully Random (no RL training)")
print(f"  Episodes Run         : {NUM_EPISODES}")
print(f"  Successful Episodes  : {num_successes} / {NUM_EPISODES}")
print(f"  Success Rate         : {success_rate:.1f}%")
print(f"  Total Reward Earned  : {total_reward_all:.1f}")
print(f"  Avg Reward/Episode   : {total_reward_all/NUM_EPISODES:.2f}")
print()
print("  Episode-by-episode breakdown:")
print(f"  {'Episode':>9}  {'Reward':>8}  {'Result':>10}  {'Bar'}")
print(f"  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*20}")
for i, (r, s) in enumerate(zip(episode_rewards, episode_success), 1):
    bar    = "#" * int(r * 20)  # filled bar if reward=1, empty if reward=0
    status = "SUCCESS" if s else "FAILED "
    print(f"  {i:>9}  {r:>8.1f}  {status:>10}  {bar if bar else '(no reward)'}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Why FrozenLake is Challenging for RL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("       WHY FROZENLAKE IS CHALLENGING FOR RL")
print("=" * 62)

challenges = [
    (
        "1. SPARSE REWARDS",
        [
            "The agent receives reward=1 ONLY when it reaches the Goal (G).",
            "Every other transition -- safe tiles, holes, mid-path steps --",
            "yields reward=0. The agent can wander for hundreds of steps",
            "without receiving any learning signal whatsoever.",
            "=> Makes it very hard to identify which actions led to success.",
        ]
    ),
    (
        "2. STOCHASTIC TRANSITIONS  (is_slippery=True)",
        [
            "Even when the agent picks action RIGHT, the environment only",
            "moves RIGHT with probability 1/3. It may slip UP or DOWN with",
            "probability 1/3 each -- completely outside the agent's control.",
            "=> The agent cannot simply memorize a fixed deterministic path.",
            "=> The optimal policy must be robust to uncontrollable slipping.",
        ]
    ),
    (
        "3. CREDIT ASSIGNMENT PROBLEM",
        [
            "When the agent receives a reward (or falls into a hole) after",
            "10-30 steps, it must figure out WHICH past actions caused the",
            "outcome. With sparse rewards and stochastic dynamics this is",
            "extremely difficult and requires many episodes of experience.",
            "=> Algorithms like Q-learning need careful discount factor (gamma)",
            "   and learning rate (alpha) tuning to back-propagate the reward",
            "   signal through long action sequences.",
        ]
    ),
    (
        "4. EXPLORATION vs. EXPLOITATION DILEMMA",
        [
            "Because success is so rare, the agent must explore aggressively",
            "early on (high epsilon in epsilon-greedy) to ever discover the",
            "goal reward. But excessive exploration wastes episodes.",
            "=> Careful epsilon-decay schedules are critical for convergence.",
            "=> This dilemma is even sharper in quantum RL where the action",
            "   selection mechanism differs from classical approaches.",
        ]
    ),
]

for title, points in challenges:
    print(f"\n  [{title}]")
    for point in points:
        print(f"      {point}")

print()
print("  +" + "-" * 58 + "+")
print("  | Summary: FrozenLake strikes a perfect balance of           |")
print("  | simplicity (4x4 grid, 4 actions) and genuine difficulty    |")
print("  | (sparse + stochastic), making it an ideal benchmark for    |")
print("  | comparing classical vs. quantum RL algorithms.             |")
print("  +" + "-" * 58 + "+")
print()
print("=" * 62)
print("  Environment setup complete! Ready for Phase 2: RL Training.")
print("=" * 62)

# ─────────────────────────────────────────────────────────────
# Clean up
# ─────────────────────────────────────────────────────────────
env.close()
