import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('dark_background')

# ── Data ──
methods_short = [
    "Q-Learning", "PPO", "Quantum Hybrid DQN",
    "DQN", "REINFORCE", "Pure VQC"
]
methods_bar = [
    "Q-Learning\n(Tabular)", "PPO\n(Deep RL)",
    "Quantum Hybrid\nDQN (Kaggle)", "DQN\n(Classical)",
    "REINFORCE\n(Policy Grad)", "Pure VQC\n(Quantum)"
]
success_rates = [72.3, 73.6, 74.4, 65.7, 0.0, 0.0]
params        = [64, 10821, 216, 5508, 10692, 36]
types         = ["Tabular TD", "Actor-Critic", "Hybrid TD",
                 "Neural TD", "MC Policy Grad", "Quantum TD"]
is_quantum    = ["No", "No", "Yes", "No", "No", "Yes"]
colors        = ["#2a9d8f", "#f72585", "#b5179e",
                 "#e9c46a", "#4a4e69", "#7209b7"]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    "Quantum RL vs Classical RL — FrozenLake-v1 Complete Comparison",
    fontsize=18, fontweight="bold", y=0.98
)

# ── Subplot 1: Success Rate Bar Chart ──
ax1 = axes[0, 0]
bars = ax1.bar(methods_bar, success_rates, color=colors,
               edgecolor='white', linewidth=1)
ax1.axhline(50, color='white', linestyle='--', alpha=0.5,
            linewidth=1.5, label="50% target")
for bar, rate in zip(bars, success_rates):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1.5,
             f"{rate:.1f}%",
             ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='white')
ax1.set_title("1. Evaluation Success Rate (Higher is Better)",
              fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Success Rate (%)", fontsize=12)
ax1.set_ylim(0, 88)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.2)
ax1.tick_params(axis='x', labelsize=9)

# ── Subplot 2: Parameter Count Horizontal Bar ──
ax2 = axes[0, 1]
sorted_idx     = np.argsort(params)
sorted_names   = [methods_short[i] for i in sorted_idx]
sorted_params  = [params[i] for i in sorted_idx]
sorted_colors  = [colors[i] for i in sorted_idx]

y_pos = np.arange(len(sorted_names))
ax2.barh(y_pos, sorted_params, color=sorted_colors,
         edgecolor='white', linewidth=1)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(sorted_names, fontsize=11)
ax2.set_xscale('log')
ax2.set_title("2. Total Parameter Count (Log Scale)",
              fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Number of Parameters (Log)", fontsize=12)
ax2.grid(axis='x', alpha=0.2, which='both')

for i, (name, p) in enumerate(zip(sorted_names, sorted_params)):
    ax2.text(p * 1.3, i, f"{p:,}",
             va='center', fontsize=10, color='white')
    if name == "Quantum Hybrid DQN":
        ax2.annotate(
            "96% fewer params\nthan Classical DQN",
            xy=(p, i), xytext=(p * 15, i + 1.2),
            arrowprops=dict(arrowstyle="->", color='#b5179e', lw=1.5),
            fontsize=10, color='#b5179e', fontweight='bold'
        )

# ── Subplot 3: Scatter — Accuracy vs Params ──
ax3 = axes[1, 0]
for i in range(len(methods_short)):
    ax3.scatter(params[i], success_rates[i],
                color=colors[i], s=280,
                edgecolor='white', linewidth=1.5, zorder=3)

# Labels for non-special points
label_offsets = {
    "Q-Learning"  : (1.4,  3,  'left'),
    "PPO"         : (1.3,  3,  'left'),
    "DQN"         : (1.3, -6,  'left'),
    "REINFORCE"   : (0.7, -7,  'right'),
    "Pure VQC"    : (1.4,  4,  'left'),
}
for i, name in enumerate(methods_short):
    if name == "Quantum Hybrid DQN":
        continue
    ox, oy, ha = label_offsets.get(name, (1.3, 3, 'left'))
    ax3.text(params[i] * ox, success_rates[i] + oy,
             name, fontsize=9, ha=ha, color='white', zorder=4)

# Special annotation for Quantum Hybrid
qh_idx = methods_short.index("Quantum Hybrid DQN")
ax3.annotate(
    "Quantum Hybrid DQN\n(Best Efficiency)",
    xy=(params[qh_idx], success_rates[qh_idx]),
    xytext=(params[qh_idx] * 6, success_rates[qh_idx] - 18),
    arrowprops=dict(arrowstyle="->", color='#b5179e', lw=2),
    fontsize=11, color='#b5179e', fontweight='bold', zorder=5
)

ax3.set_xscale('log')
ax3.set_title("3. Efficiency: Success Rate vs Model Size",
              fontsize=13, fontweight="bold", pad=12)
ax3.set_xlabel("Total Parameters (Log Scale)", fontsize=12)
ax3.set_ylabel("Success Rate (%)", fontsize=12)
ax3.set_ylim(-15, 88)
ax3.set_xlim(20, 30000)
ax3.grid(alpha=0.2, which='both', linestyle='--')

# ── Subplot 4: Summary Table ──
ax4 = axes[1, 1]
ax4.axis('off')

columns   = ["Method", "Success %", "Params", "Type", "Quantum?"]
order_idx = np.argsort(success_rates)[::-1]
table_data = []
for i in order_idx:
    table_data.append([
        methods_short[i],
        f"{success_rates[i]:.1f}%",
        f"{params[i]:,}",
        types[i],
        is_quantum[i]
    ])

table = ax4.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.4)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#555555')
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#222222')
    elif table_data[row-1][0] == "Quantum Hybrid DQN":
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4a154b')
    else:
        cell.set_facecolor('#111111')
        cell.set_text_props(color='white')

# Widen Method column
table.auto_set_column_width([0, 1, 2, 3, 4])

ax4.set_title("4. Aggregate Performance Summary",
              fontsize=13, fontweight="bold", pad=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("comparison_plot.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved → comparison_plot.png")
