import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Load the CSV file
csv_file = "/home/lenman/capstone/parallelrm/src/vectors.xlsx"  # Replace with your file path
df = pd.read_excel(csv_file)

# Extract columns
x = df["x"]
y = df["y"]
neighbors = df[["n1", "n2", "n3"]]

# Create a graph
G = nx.Graph()

# Add nodes
for i in range(len(df)):
    G.add_node(i, pos=(x[i], y[i]))

# Add edges (connections to nearest neighbors)
for i, row in neighbors.iterrows():
    for neighbor in row:
        G.add_edge(i, neighbor)

# Get node positions
pos = nx.get_node_attributes(G, "pos")

# Plot the graph
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_color="blue", edge_color="gray", node_size=100)
plt.title("kNN Graph Visualization")
plt.show()
