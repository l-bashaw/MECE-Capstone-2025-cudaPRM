import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# === CONFIG ===
FOLDER = "exp_results"

# Academic figure styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
    'patch.linewidth': 1.5,
})

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# === DATA LOADING ===
def load_data(folder):
    data = {}
    for filename in os.listdir(folder):
        if filename.endswith(".json") and "comparison_experiment_trajectories" in filename:
            match = re.search(r"trajectories_(\d+)\.json", filename)
            if not match:
                continue
            nodes = int(match.group(1))
            with open(os.path.join(folder, filename), "r") as f:
                content = json.load(f)

            timings = []
            perc_sums = []
            path_lengths = []
            boxplots = {"whislo": [], "q1": [], "med": [], "q3": [], "whishi": []}

            for scenario, runs in content.items():
                for run in runs:
                    timings.append(run["timing"]["mean"])
                    perc_sums.append(run["perc_score"]["sum"])
                    path_lengths.append(run["path_length"])
                    b = run["boxplot"]
                    for k in boxplots.keys():
                        boxplots[k].append(b[k])

            data[nodes] = {
                "timing_means": timings,
                "perc_sum_mean": np.mean(perc_sums),
                "path_length_mean": np.mean(path_lengths),
                "boxplot": {k: np.mean(v) for k, v in boxplots.items()}
            }
    return dict(sorted(data.items()))


# === SUMMARY PRINT ===
def create_summary_stats(data):
    nodes = list(data.keys())
    timings = [np.mean(data[n]["timing_means"]) * 1000 for n in nodes]  # ms

    print("\n" + "="*50)
    print("PRM PERFORMANCE ANALYSIS SUMMARY")
    print("="*50)

    for node, timing in zip(nodes, timings):
        complexity_level = "Low" if timing < 50 else "Medium" if timing < 100 else "High"
        print(f"Nodes: {node:5,} | Timing: {timing:6.1f} ms | Complexity: {complexity_level}")

    print("="*50)
    print(f"Performance Range: {min(timings):.1f} ms - {max(timings):.1f} ms")
    print("="*50 + "\n")


# === PLOTTING ===
def plot_dual_axis(data):
    nodes = sorted(data.keys())

    # Prepare timing boxplot data
    box_data = []
    for n in nodes:
        b = data[n]["boxplot"]
        box_data.append({
            "label": str(n),
            "whislo": b["whislo"],   # low whisker
            "q1": b["q1"],           # 1st quartile
            "med": b["med"],         # median
            "q3": b["q3"],           # 3rd quartile
            "whishi": b["whishi"],   # high whisker
            "fliers": []             # omit outliers
        })

    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=600)
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    # --- Left axis: Timing (boxplots) ---
    widths = [500 for n in nodes]  # scale widths relative to node value
    bp = ax1.bxp(box_data, positions=nodes, widths=widths,
                 patch_artist=True, showfliers=False)

    for patch in bp['boxes']:
        patch.set(facecolor="#EDC824", alpha=0.6, edgecolor="#000000")

    for prop in ['whiskers', 'caps', 'medians']:
        for line in bp[prop]:
            line.set(color='#333333', linewidth=1.5)

    ax1.set_xlabel("Number of Nodes in Roadmap", fontsize=18, labelpad=15)
    ax1.set_ylabel("PRM Build and Query Time (s)", fontsize=18, color='black', labelpad=15)
    ax1.set_xscale("linear")
    ax1.set_yscale("log")
    ax1.set_ylim(top=1)  # Set y-axis limit to 10^0 = 1
    # Increase tick font size
    ax1.tick_params(axis='x', labelsize=14, pad=10)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=14, pad=10)
    ax1.grid(True, which="both", linestyle='-', alpha=0.25)
   
    # --- Right axis: Perception Sum + Path Length (means only) ---
    perc_means = [data[n]["perc_sum_mean"] for n in nodes]
    path_means = [data[n]["path_length_mean"] for n in nodes]

    # ax2 = ax1.twinx()
    # ax2.plot(nodes, perc_means, '-s', color='tab:green', label="Perception Sum")
    # ax2.plot(nodes, path_means, '-^', color='tab:red', label="Path Length")
    # ax2.set_ylabel("Perception Sum / Path Length", fontsize=14, color='tab:red')
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    # # Legends
    # lines, labels = ax2.get_legend_handles_labels()
    # ax2.legend(lines, labels, loc='upper left')

    # plt.title("PRM Planner Performance Metrics vs Roadmap Size", fontsize=16, fontweight='bold')
    # plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "dual_axis_performance_boxplot__.pdf"),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()


# === MAIN ===
if __name__ == "__main__":
    data = load_data(FOLDER)
    create_summary_stats(data)
    plot_dual_axis(data)









# import os
# import re
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # Set the aesthetic style for academic papers
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'DejaVu Serif'],
#     'font.size': 11,
#     'axes.linewidth': 1.2,
#     'grid.linewidth': 0.8,
#     'lines.linewidth': 2,
#     'patch.linewidth': 1.5
# })

# FOLDER = "exp_results"

# def load_data(folder):
#     data = {}
#     for filename in os.listdir(folder):
#         if filename.endswith(".json") and "comparison_experiment_trajectories" in filename:
#             # extract number of nodes from filename
#             match = re.search(r"trajectories_(\d+)\.json", filename)
#             if not match:
#                 continue
#             nodes = int(match.group(1))
#             with open(os.path.join(folder, filename), "r") as f:
#                 content = json.load(f)
#             timings = []
#             perc_scores = []
#             boxplots = {"whislo": [], "q1": [], "med": [], "q3": [], "whishi": []}
#             # Each top-level key (e.g., "human", "monitor") is a scenario
#             for scenario, runs in content.items():
#                 for run in runs:
#                     t = run["timing"]
#                     s = run["perc_score"]
#                     b = run["boxplot"]
#                     timings.append(t["mean"])
#                     perc_scores.append(s["mean"])
#                     for k in boxplots.keys():
#                         boxplots[k].append(b[k])
#             # aggregate
#             data[nodes] = {
#                 "timing_mean": np.mean(timings),
#                 "timing_std": np.std(timings),
#                 "perc_score_mean": np.mean(perc_scores),
#                 "perc_score_std": np.std(perc_scores),
#                 "boxplot": {k: np.mean(v) for k, v in boxplots.items()}
#             }
#     return dict(sorted(data.items()))

# import matplotlib.ticker as ticker

# def plot_mean_with_errorbars(data, num_std_devs=1.0):
#     """
#     Plot mean values with error bars instead of boxplots
#     with logarithmic X-axis and scientific Y-axis in seconds.
#     """
#     fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
#     fig.patch.set_facecolor('white')
#     ax.set_facecolor('white')
    
#     nodes = sorted(data.keys())
#     means = []
#     errors = []
    
#     # Extract means and standard deviations, convert to SECONDS
#     for n in nodes:
#         mean_s = data[n]["timing_mean"]   # already in seconds
#         std_s = data[n]["timing_std"]
#         means.append(mean_s)
#         errors.append(std_s * num_std_devs)
    
#     # Academic color palette with a bit of variation
#     colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(nodes)))
    
#     # Plot with error bars
#     for i, n in enumerate(nodes):
#         ax.errorbar(n, means[i], yerr=errors[i],
#                     fmt='o',
#                     color=colors[i],
#                     ecolor=colors[i],
#                     elinewidth=1.8,
#                     capsize=5,
#                     capthick=1.4,
#                     markersize=8,
#                     markerfacecolor=colors[i],
#                     markeredgecolor='white',
#                     markeredgewidth=1.2,
#                     alpha=0.9,
#                     zorder=3)
    
#     # Connect with line (gradient look using single color for clarity)
#     ax.plot(nodes, means, '-', color='#2E5F88', alpha=0.6, linewidth=2, zorder=2)
    
#     # Logarithmic X-axis
#     ax.set_xscale("log")
    
#     # Y-axis in scientific notation (seconds)
#     ax.set_yscale("log")
#     ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))
    
#     # Gridlines (faint, both directions)
#     ax.grid(True, which="both", linestyle='-', alpha=0.25, color='#AAAAAA', linewidth=0.7)
#     ax.set_axisbelow(True)
    
#     # Axis styling
#     for spine in ax.spines.values():
#         spine.set_color('#333333')
#         spine.set_linewidth(1.1)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Labels
#     ax.set_xlabel("Number of Nodes in Roadmap (log scale)", fontsize=14, color='#333333', labelpad=12)
#     ax.set_ylabel("Timing (s)", fontsize=14, color='#333333', labelpad=12)
    
#     # Title
#     error_bar_text = f"±{num_std_devs}σ" if num_std_devs != 1.0 else "±1σ"
#     ax.set_title(f"PRM Timing Performance vs Roadmap Size\n(Mean {error_bar_text} Error Bars)", 
#                  fontsize=16, fontweight='bold', color='#333333', pad=20)
    
#     # Tick styling
#     ax.tick_params(colors='#333333', labelsize=11, width=1.0, length=5)
    
#     plt.tight_layout()
    
#     # Save
#     filename_suffix = f"_errorbar_{num_std_devs}std" if num_std_devs != 1.0 else "_errorbar"
#     plt.savefig(os.path.join(FOLDER, f"timing_vs_nodes{filename_suffix}_log.png"), 
#                 dpi=300, bbox_inches='tight', facecolor='white',
#                 edgecolor='none', pad_inches=0.1)
#     plt.show()

# def plot_boxplots(data):
#     """Legacy boxplot function - kept for compatibility"""
#     # Create figure with clean academic styling
#     fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
#     fig.patch.set_facecolor('white')
#     ax.set_facecolor('white')
    
#     nodes = sorted(data.keys())
#     box_data = []
    
#     for n in nodes:
#         b = data[n]["boxplot"]
#         box_data.append({
#             "whislo": b["whislo"] * 1000,
#             "q1": b["q1"] * 1000,
#             "med": b["med"] * 1000,
#             "q3": b["q3"] * 1000,
#             "whishi": b["whishi"] * 1000,
#             "fliers": []
#         })
    
#     # Elegant academic color palette - sophisticated blues and grays
#     base_color = '#2E5F88'  # Professional blue
#     colors = [plt.cm.Blues(0.4 + 0.4 * i / (len(nodes) - 1)) for i in range(len(nodes))]
    
#     # Create clean, elegant boxplots
#     bp = ax.bxp(box_data, positions=nodes, widths=350, 
#                 patch_artist=True, showfliers=False,
#                 whiskerprops=dict(color=base_color, linewidth=1.8, alpha=0.8),
#                 capprops=dict(color=base_color, linewidth=1.8, alpha=0.8),
#                 medianprops=dict(color='#B8860B', linewidth=2.5))  # Elegant gold for median
    
#     # Style each box with sophisticated colors
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.7)
#         patch.set_edgecolor(base_color)
#         patch.set_linewidth(1.2)
    
#     # Minimal, clean grid
#     ax.grid(True, axis="y", linestyle='-', alpha=0.3, color='#CCCCCC', linewidth=0.8)
#     ax.set_axisbelow(True)
    
#     # Clean axis styling
#     for spine in ax.spines.values():
#         spine.set_color('#333333')
#         spine.set_linewidth(1.2)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Professional typography
#     ax.set_xlabel("Number of Nodes in Roadmap", fontsize=14, fontweight='normal', 
#                   color='#333333', labelpad=12)
#     ax.set_ylabel("Timing (ms)", fontsize=14, fontweight='normal', 
#                   color='#333333', labelpad=12)
    
#     # Clean, academic title
#     ax.set_title("PRM Timing Performance vs Roadmap Size", 
#                 fontsize=16, fontweight='bold', color='#333333', pad=20)
    
#     # Clean tick styling
#     ax.tick_params(colors='#333333', labelsize=11, width=1.2, length=5)
#     ax.set_xticks(nodes)
#     ax.set_xticklabels([f"{n:,}" for n in nodes], rotation=45, ha='right')
    
#     # Format y-axis labels properly
#     y_ticks = ax.get_yticks()
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels([f"{int(y):,}" for y in y_ticks])
    
#     # Set limits with appropriate padding
#     ax.set_xlim(min(nodes) - 500, max(nodes) + 500)
    
#     plt.tight_layout()
    
#     # Save with publication quality
#     plt.savefig(os.path.join(FOLDER, "timing_vs_nodes_boxplot_academic.png"), 
#                 dpi=300, bbox_inches='tight', facecolor='white', 
#                 edgecolor='none', pad_inches=0.1)
#     plt.show()

# def create_summary_stats(data):
#     """Create a clean summary statistics display"""
#     nodes = list(data.keys())
#     timings = [data[n]["timing_mean"] * 1000 for n in nodes]  # Use timing_mean for error bar plot
    
#     print("\n" + "="*50)
#     print("PRM PERFORMANCE ANALYSIS SUMMARY")
#     print("="*50)
    
#     for node, timing in zip(nodes, timings):
#         complexity_level = "Low" if timing < 50 else "Medium" if timing < 100 else "High"
#         print(f"Nodes: {node:5,} | Timing: {timing:6.1f}ms | Complexity: {complexity_level}")
    
#     print("="*50)
#     print(f"Performance Range: {min(timings):.1f}ms - {max(timings):.1f}ms")
#     print(f"Average Scaling: {(max(timings) - min(timings)) / (max(nodes) - min(nodes)) * 1000:.2f}ms per 1K nodes")
#     print("="*50 + "\n")

# if __name__ == "__main__":
#     data = load_data(FOLDER)
#     create_summary_stats(data)
    
#     # Plot mean with error bars (adjustable standard deviations)
#     # Options: 1.0 (±1σ), 1.96 (±1.96σ ≈ 95% CI), 2.0 (±2σ), etc.
#     std_devs = 1  # Change this value to adjust error bar size
#     plot_mean_with_errorbars(data, num_std_devs=std_devs)
    
#     # Uncomment below to also generate boxplots
#     # plot_boxplots(data)

# import os
# import re
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.colors import LinearSegmentedColormap
# import seaborn as sns

# # Set the aesthetic style
# plt.style.use('dark_background')
# sns.set_palette("husl")

# FOLDER = "exp_results"

# def load_data(folder):
#     data = {}
#     for filename in os.listdir(folder):
#         if filename.endswith(".json") and "comparison_experiment_trajectories" in filename:
#             # extract number of nodes from filename
#             match = re.search(r"trajectories_(\d+)\.json", filename)
#             if not match:
#                 continue
#             nodes = int(match.group(1))
#             with open(os.path.join(folder, filename), "r") as f:
#                 content = json.load(f)
#             timings = []
#             perc_scores = []
#             boxplots = {"whislo": [], "q1": [], "med": [], "q3": [], "whishi": []}
#             # Each top-level key (e.g., "human", "monitor") is a scenario
#             for scenario, runs in content.items():
#                 for run in runs:
#                     t = run["timing"]
#                     s = run["perc_score"]
#                     b = run["boxplot"]
#                     timings.append(t["mean"])
#                     perc_scores.append(s["mean"])
#                     for k in boxplots.keys():
#                         boxplots[k].append(b[k])
#             # aggregate
#             data[nodes] = {
#                 "timing_mean": np.mean(timings),
#                 "timing_std": np.std(timings),
#                 "perc_score_mean": np.mean(perc_scores),
#                 "perc_score_std": np.std(perc_scores),
#                 "boxplot": {k: np.mean(v) for k, v in boxplots.items()}
#             }
#     return dict(sorted(data.items()))

# def plot_boxplots(data):
#     # Create figure with high DPI for crisp output
#     fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
#     fig.patch.set_facecolor('#0a0a0a')
#     ax.set_facecolor('#111111')
    
#     nodes = sorted(data.keys())
#     box_data = []
    
#     for n in nodes:
#         b = data[n]["boxplot"]
#         box_data.append({
#             "whislo": b["whislo"] * 1000,
#             "q1": b["q1"] * 1000,
#             "med": b["med"] * 1000,
#             "q3": b["q3"] * 1000,
#             "whishi": b["whishi"] * 1000,
#             "fliers": []
#         })
    
#     # Create beautiful gradient colors
#     colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(nodes)))
    
#     # Create boxplots with custom styling
#     bp = ax.bxp(box_data, positions=nodes, widths=400, 
#                 patch_artist=True, showfliers=False,
#                 whiskerprops=dict(color='white', linewidth=2.5, alpha=0.8),
#                 capprops=dict(color='white', linewidth=2.5, alpha=0.8),
#                 medianprops=dict(color='#FFD700', linewidth=4, alpha=0.9))
    
#     # Style each box with gradient colors and glow effect
#     for patch, color, node, box_info in zip(bp['boxes'], colors, nodes, box_data):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.8)
#         patch.set_edgecolor('white')
#         patch.set_linewidth(2)
        
#         # Add subtle glow effect using box data
#         box_height = box_info['q3'] - box_info['q1']
#         box_bottom = box_info['q1']
#         glow = mpatches.Rectangle((node - 200, box_bottom - box_height * 0.1), 400, box_height * 1.2,
#                                 facecolor=color, alpha=0.3, zorder=0)
#         ax.add_patch(glow)
    
#     # Create stunning grid
#     ax.grid(True, axis="y", linestyle='-', alpha=0.2, color='white', linewidth=0.8)
#     ax.grid(True, axis="x", linestyle='-', alpha=0.1, color='white', linewidth=0.5)
    
#     # Beautiful axis styling
#     ax.spines['bottom'].set_color('#444444')
#     ax.spines['left'].set_color('#444444')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(2)
#     ax.spines['left'].set_linewidth(2)
    
#     # Enhanced labels and title
#     ax.set_xlabel("Number of Nodes in Roadmap", fontsize=16, fontweight='bold', 
#                   color='white', labelpad=15)
#     ax.set_ylabel("Timing (ms)", fontsize=16, fontweight='bold', 
#                   color='white', labelpad=15)
    
#     # Stunning title with gradient effect
#     title = ax.set_title("PRM Performance Analysis\nTiming Distribution vs Roadmap Complexity", 
#                         fontsize=22, fontweight='bold', color='white', pad=30)
    
#     # Beautiful tick styling
#     ax.tick_params(colors='white', labelsize=12, width=2, length=6)
#     ax.set_xticks(nodes)
#     ax.set_xticklabels([f"{n:,}" for n in nodes], rotation=45, ha='right')
    
#     # Format y-axis labels
#     y_ticks = ax.get_yticks()
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels([f"{int(y):,}" for y in y_ticks])
    
#     # Set limits with padding
#     ax.set_xlim(min(nodes) - 600, max(nodes) + 600)
    
#     # Add subtle background elements
#     y_min, y_max = ax.get_ylim()
#     for i, node in enumerate(nodes[::2]):  # Every other node
#         ax.axvspan(node - 300, node + 300, alpha=0.05, color='white', zorder=0)
    
#     # Add performance insight annotation
#     max_timing_node = max(nodes, key=lambda n: data[n]["boxplot"]["med"])
#     max_timing = data[max_timing_node]["boxplot"]["med"] * 1000
    
#     ax.annotate(f'Peak Performance\nCost: {max_timing:.1f}ms', 
#                 xy=(max_timing_node, max_timing), 
#                 xytext=(max_timing_node + 1000, max_timing + (y_max - y_min) * 0.2),
#                 fontsize=11, color='#FFD700', fontweight='bold',
#                 arrowprops=dict(arrowstyle='->', color='#FFD700', alpha=0.8, lw=2),
#                 bbox=dict(boxstyle="round,pad=0.3", facecolor='black', 
#                          edgecolor='#FFD700', alpha=0.8))
    
#     # Add subtle watermark
#     fig.text(0.99, 0.01, 'PRM Analysis Dashboard', 
#              fontsize=8, color='gray', alpha=0.5, 
#              ha='right', va='bottom', style='italic')
    
#     plt.tight_layout()
    
#     # Save with high quality
#     plt.savefig(os.path.join(FOLDER, "timing_vs_nodes_boxplot_beautiful.png"), 
#                 dpi=300, bbox_inches='tight', facecolor='#0a0a0a', 
#                 edgecolor='none', pad_inches=0.2)
#     plt.show()

# def create_summary_stats(data):
#     """Create a beautiful summary statistics overlay"""
#     nodes = list(data.keys())
#     timings = [data[n]["boxplot"]["med"] * 1000 for n in nodes]
    
#     print("\n" + "="*60)
#     print("🚀 PRM PERFORMANCE ANALYSIS SUMMARY")
#     print("="*60)
    
#     for node, timing in zip(nodes, timings):
#         complexity_level = "🟢 Low" if timing < 50 else "🟡 Medium" if timing < 100 else "🔴 High"
#         print(f"Nodes: {node:5,} | Timing: {timing:6.1f}ms | Complexity: {complexity_level}")
    
#     print("="*60)
#     print(f"Performance Range: {min(timings):.1f}ms - {max(timings):.1f}ms")
#     print(f"Average Scaling: {(max(timings) - min(timings)) / (max(nodes) - min(nodes)) * 1000:.2f}ms per 1K nodes")
#     print("="*60 + "\n")

# if __name__ == "__main__":
#     data = load_data(FOLDER)
#     create_summary_stats(data)
#     plot_boxplots(data)