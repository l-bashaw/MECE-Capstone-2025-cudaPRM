import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Directory containing the timing files
directory = './timing_results'  # change this to your actual directory

# Regex to extract number of states and K
pattern = re.compile(r"times_(\d+)_states_(\d+)_K\.txt")



data = []

# Parse files
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        num_states = int(match.group(1))
        k = int(match.group(2))
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()[1:]  # skip header
            times = np.array([float(line.strip()) for line in lines])
            mean_time = np.mean(times[1:]) # skip first time (it includes warmup overhead)
            data.append((num_states, k, mean_time))

# Create DataFrame
df = pd.DataFrame(data, columns=["States", "K", "MeanTime"])

# Create pivot table for 3D plotting
pivot = df.pivot(index="States", columns="K", values="MeanTime")
states = pivot.index.values
ks = pivot.columns.values
times = pivot.values


# Flatten the data for interpolation
points = np.array([(k, s) for s in states for k in ks])
values = times.flatten()

# Create a finer grid
k_fine = np.linspace(min(ks), max(ks), 6)
s_fine = np.linspace(min(states), max(states), 6)
K_fine, S_fine = np.meshgrid(k_fine, s_fine)

# Interpolate
Z_fine = griddata(points, values, (K_fine, S_fine), method='cubic')

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(K_fine, S_fine, Z_fine, cmap='viridis', edgecolor='none', alpha=0.95)

ax.set_xlabel("Number of Nearest Neighbors", fontsize=16)
ax.set_ylabel("Number of States", fontsize=16)
ax.set_zlabel("Time (ms)", fontsize=16)
#ax.set_title("Graph Generation Time for a 4 Obstacle Environment", fontsize=18)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# give more spacing between state axis label and ticks
ax.yaxis.set_tick_params(pad=0)
ax.set_zlim(0, 9)

# set azimuth and elevation
ax.view_init(elev=14, azim=-147)

plt.tight_layout()
plt.savefig("time_surface.jpg", dpi=1000)
plt.show()

# # Plot 2D Line Plot: Time vs Number of States (Shading with K)
# plt.figure(figsize=(10, 6))

# for i, k in enumerate(ks):
#     alpha = 1.0 if k == min(ks) else 0.3 + 0.7 * (1 - (k - min(ks)) / (max(ks) - min(ks)))  # fade with K
#     label = f'K={k}' 
#     plt.plot(states, times[:, i], label=label, color=plt.cm.viridis(i / len(ks)))  # Use color map for different lines
#     if i < len(ks) - 1:  # Fade the region between lines
#         plt.fill_between(states, times[:, i], times[:, i + 1], alpha=0.1 + 0.3 * alpha, color=plt.cm.viridis(i / len(ks)))

# plt.title("Time vs Number of States (Shading with Higher K)")
# plt.xlabel("Number of States")
# plt.ylabel("Time (ms)")
# plt.legend()
# plt.grid(True)
# #plt.tight_layout()
# plt.show()

# # Plot 2D Line Plot: Time vs K (Shading with States)
# plt.figure(figsize=(10, 6))

# for i, state in enumerate(states):
#     # Fading effect for states in the Y-axis (fading lines for each state)
#     alpha = 1.0 if state == max(states) else 0.3 + 0.7 * (state - min(states)) / (max(states) - min(states))  # fade with states
#     label = f'States={state}' 
#     plt.plot(ks, times[i, :], label=label, color=plt.cm.viridis(i / len(states)))  # Use color map for different lines
#     if i < len(states) - 1:  # Fade the region between lines
#         plt.fill_between(ks, times[i, :], times[i + 1, :], alpha=0.1 + 0.3 * alpha, color=plt.cm.viridis(i / len(states)))

# plt.title("Time vs K (Shading with Higher Number of States)")
# plt.xlabel("K (Nearest Neighbors)")
# plt.ylabel("Time (ms)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()