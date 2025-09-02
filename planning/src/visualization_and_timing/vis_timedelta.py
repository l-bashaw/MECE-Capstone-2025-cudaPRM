import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def load_mean_times(directory):
    pattern = re.compile(r"times_(\d+)_states_(\d+)_K\.txt")
    data = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            num_states = int(match.group(1))
            k = int(match.group(2))
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()[1:]  # skip header
                times = np.array([float(line.strip()) for line in lines])
                mean_time = np.mean(times[1:])
                data.append((num_states, k, mean_time))
    
    df = pd.DataFrame(data, columns=["States", "K", "MeanTime"])
    return df

# Load data
df1 = load_mean_times("timing_results")
df2 = load_mean_times("timing_results_env2")

# Merge on States and K
merged = pd.merge(df1, df2, on=["States", "K"], suffixes=("_env1", "_env2"))
merged["TimeDiff"] = merged["MeanTime_env2"] - merged["MeanTime_env1"]

# Create pivot table
pivot = merged.pivot(index="States", columns="K", values="TimeDiff")
states = pivot.index.values
ks = pivot.columns.values
diffs = pivot.values

# Flatten and interpolate
points = np.array([(k, s) for s in states for k in ks])
values = diffs.flatten()
k_fine = np.linspace(min(ks), max(ks), 6)
s_fine = np.linspace(min(states), max(states), 6)
K_fine, S_fine = np.meshgrid(k_fine, s_fine)
Z_fine = griddata(points, values, (K_fine, S_fine), method='cubic')

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(K_fine, S_fine, Z_fine, cmap='coolwarm', edgecolor='none', alpha=0.95)

ax.set_xlabel("Number of Nearest Neighbors", fontsize=16)
ax.set_ylabel("Number of States", fontsize=16)
ax.set_zlabel("Time Delta (ms)", fontsize=16)
ax.set_title("Graph Construction Time for 16 vs. 4 Obstacles", fontsize=20)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

ax.yaxis.set_tick_params(pad=0)
ax.view_init(elev=14, azim=-147)

plt.tight_layout()
plt.savefig("timedelta_surface.jpg", dpi=1000)

plt.show()
