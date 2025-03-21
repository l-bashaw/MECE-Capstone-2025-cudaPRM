import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file (assuming two columns: x and y)
poses = np.loadtxt("capstone/parallelrm/posesImproved.csv", delimiter=",")


# Extract x and y coordinates
x = poses[:, 0]
y = poses[:, 1]

# Create a scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=10, alpha=0.6, c="blue", edgecolors="k", label="2D Poses")

# Add grid and labels
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Scatter Plot of 2D Poses")
plt.legend()
plt.axis("equal")  # Ensure equal scaling for x and y

# Show the plot
plt.show()

times = np.loadtxt("capstone/parallelrm/times_7_10k_ms.csv", delimiter=",")[1:]
times = times # Convert to seconds

# Calculate statistical values
q1 = np.percentile(times, 25)
q3 = np.percentile(times, 75)
mean = np.mean(times)

# Create the boxplot with improved aesthetics
plt.figure(figsize=(8, 4))
sns.boxplot(x=times, orient='h', color='skyblue')

# Set the x-axis to a log scale
plt.xscale('log')

# Add the legend with mean, Q1, and Q3
plt.text(0.7, 0.9, f'Mean: {mean:.4f} (ms)', transform=plt.gca().transAxes, fontsize=12, color='black')
plt.text(0.7, 0.85, f'Q1: {q1:.4f} (ms)', transform=plt.gca().transAxes, fontsize=12, color='black')
plt.text(0.7, 0.8, f'Q3: {q3:.4f} (ms)', transform=plt.gca().transAxes, fontsize=12, color='black')

# Improve labels and appearance
plt.xlabel("Time (log scale)", fontsize=12)
plt.title("Boxplot of Times", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()