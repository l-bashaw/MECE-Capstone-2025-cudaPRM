import matplotlib.pyplot as plt
import numpy as np

hi_load = {
    "State Generation": 0.1353,
    "Nearest Neighbors": 9.71,
    "Edge Construction": 0.74314,
    "Collision Checking": (0.00518+0.14768),
    "Profiling Overhead": None,
}

mid_load = {
    "State Generation": .11706,
    "Nearest Neighbors": 1.67,
    "Edge Construction": .11955,
    "Collision Checking": (.00464+0.03114),
    "Profiling Overhead": None,
}

lo_load = {
    "State Generation": 0.10045,
    "Nearest Neighbors": 0.21862,
    "Edge Construction": 0.05994,
    "Collision Checking": (0.00291+0.00771),
    "Profiling Overhead": None,
}


# Aggregate data
labels = ["State Generation", "Nearest Neighbors", "Edge Construction", "Collision Checking"]
loads = ["Case 1", "Case 2", "Case 3"]
data = [lo_load, mid_load, hi_load]

descriptions = ["N=1000, k=5, Obs=4", "N=5000, k=10, Obs=16", "N=20000, k=20, Obs=16"]

# Extract values per category
category_values = {label: [] for label in labels}
for label in labels:
    for d in data:
        category_values[label].append(d[label])

# Create stacked bar chart
x = np.arange(len(loads))
width = 0.5

fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(loads))

colors = {
    "State Generation": "#1f77b4",
    "Nearest Neighbors": "#ff7f0e",
    "Edge Construction": "#2ca02c",
    "Collision Checking": "#d62728",
}

for label in labels:
    ax.bar(x, category_values[label], width, label=label, bottom=bottom, color=colors[label])
    bottom += category_values[label]

# Customization
ax.set_ylabel("Time (ms)", fontsize=16)
ax.set_title("Graph Construction Runtime Breakdown", fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(loads, fontsize=16)

ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=16)

# Custom legend for case descriptions
from matplotlib.patches import Patch
# Phase legend (top right inside)
phase_legend = ax.legend(loc="upper left", frameon=True, fontsize=14)


# Make sure both legends show
ax.add_artist(phase_legend)

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("timing_breakdown.jpg", dpi=1000)
plt.tight_layout()
plt.show()