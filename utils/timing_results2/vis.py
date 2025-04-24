import matplotlib.pyplot as plt

# Read and parse the file
filename = "summary.txt"

batch_sizes = []
fk_times = []
nn_times = []
total_times = []

with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    if line.strip().startswith('|') and line.count('|') > 5 and "Batch Size" not in line:
        cols = line.strip().split('|')[1:-1]
        cols = [c.strip() for c in cols]
        batch_sizes.append(int(cols[0]))
        fk_times.append(float(cols[1])*1000)  # Convert to milliseconds
        nn_times.append(float(cols[2])*1000)
        total_times.append(float(cols[3]))

# Compute speedup factors (total time ratio from previous)
factors = [None]  # First element has no previous value
for i in range(1, len(total_times)):
    factors.append((total_times[i] / total_times[i - 1]) / (batch_sizes[i] / batch_sizes[i - 1]))

# Plotting
bar_width = 0.6
x = range(len(batch_sizes))  # Index-based x-axis for bar positions

fig, ax1 = plt.subplots(figsize=(14, 12))

# Primary y-axis: FK + NN Time (stacked bar)
fk_bar = ax1.bar(x, fk_times, width=bar_width, label='FK Time', color='skyblue')
nn_bar = ax1.bar(x, nn_times, width=bar_width, bottom=fk_times, label='NN Time', color='orange')

ax1.set_xlabel('Batch Size', fontsize=16)
ax1.set_ylabel('Total Time (ms)', fontsize=16)
ax1.set_title('Execution Time for State Projection and Surrogate Model Inference', fontsize=18)
ax1.set_xticks(x)
ax1.set_xticklabels(batch_sizes, fontsize=14)
ax1.legend(loc='upper left', bbox_to_anchor=(0.015, 1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Secondary y-axis: Time increase factor
ax2 = ax1.twinx()
factor_x = x[1:]  # Skip first index
factor_y = [f for f in factors[1:]]

ax2.plot(factor_x, factor_y, 'o--', color='red', label='Scale Factor')
for xi, yi in zip(factor_x, factor_y):
    ax2.text(xi, yi + 0.01, f"{yi:.2f}x", color='red', ha='right', va='bottom', fontsize=14)
ax2.set_ylabel('Time Delta/Batch Size Delta', fontsize=16)
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9))

# Set y axis fontsize
ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14) 

# Increase font for better readability
plt.show()
