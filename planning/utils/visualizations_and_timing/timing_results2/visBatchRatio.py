import matplotlib.pyplot as plt
import numpy as np

N = 10000

K = np.arange(2, 50, 1)
R = np.arange(3, 50, 1)


K_grid, R_grid = np.meshgrid(K, R, indexing='ij')

total_states = N + (N * K_grid * (R_grid - 2)) // 2

print(total_states[8][7])

# Must project all states, only have to score the nodes
states2project_over_states2score = total_states // (N)


#plot surface of total states and 
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K_grid, R_grid, total_states, cmap='viridis', edgecolor='none')
ax.set_xlabel('K')
ax.set_ylabel('R')
ax.set_zlabel('Total Number of States')
ax.set_title('Total Number of States vs K and R')
plt.show()

#plot the ratio as a surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K_grid, R_grid, states2project_over_states2score, cmap='viridis', edgecolor='none')
ax.set_xlabel('K')
ax.set_ylabel('R')
ax.set_zlabel('Total Number of States / N')
ax.set_title('Ratio of States to Project over States to Score')
plt.show()