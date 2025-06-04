# Import necessary libraries for building the model, processing data, and visualizing
import matplotlib
matplotlib.use('Agg')
import torch  # For building neural networks
import torch.nn.functional as F  # For activation functions like ReLU
from torch_geometric.nn import GATv2Conv  # GAT layer from PyTorch Geometric
import networkx as nx
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
from data_processing import load_all_graphs, split_data, save_data, load_data
import seaborn as sns
from scipy.spatial.distance import pdist, squareform


#  Data Preprocessing
data_dir = r"C:\Users\hites\Desktop\PROJECTARBEIT\architecture\Input\N20"
save_path = r"C:\Users\hites\Desktop\PROJECTARBEIT\architecture\Input\N20\processed_graph.pt"

# data_dir = r"/work/ws-tmp/g062603-GATEXP/architecture/Input/N20"
# save_path = r"/work/ws-tmp/g062603-GATEXP/architecture/Input/N20/processed_graphs.pt"

directory = os.path.dirname(os.path.abspath(__file__))

result_dir=os.path.join(directory, "results")
os.makedirs(result_dir, exist_ok=True) 

model_dir=os.path.join(directory, "model")
os.makedirs(model_dir, exist_ok=True)

fig_dir=os.path.join(directory, "figures")
os.makedirs(fig_dir, exist_ok=True)

if os.path.exists(save_path):
    print(f"Loading preprocessed data from {save_path}...")
    graph_list = load_data(save_path)
else:
    print(f"Preprocessing JSON files from {data_dir}...")
    graph_list = load_all_graphs(data_dir)
    save_data(graph_list, save_path)
    print(f"Preprocessed data saved to {save_path}.")

train_graphs, val_graphs, test_graphs = split_data(graph_list, train_count=16, val_count=3, test_count=1)

print(f"Number of training graphs: {len(train_graphs)}")
print(f"Number of validation graphs: {len(val_graphs)}")
print(f"Number of test graphs: {len(test_graphs)}")


# Define hyperparameter search space
hidden_dims = [8, 16, 32, 64]
output_dims = [4, 8]
margins = [0.5, 1.0, 1.5]
num_heads = [2, 4]
# Learning rate fixed (0.001), not tuning it.



# Define the GAT Model
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GATModel, self).__init__()
        
        # First GAT layer with edge features
        self.gat1 = GATv2Conv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True  # Use concatenation of multiple attention heads
        )
        
        # Second GAT layer with edge features
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,  # Adjust input size based on concatenated heads
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True  # Do not concatenate at the output layer
        )

        self.gat3 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=output_dim,
            heads=1,
            concat=False  # Do not concatenate at the output layer
        )

    def forward(self, x, edge_index):
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.relu(x)  # Apply activation function
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        # Third GAT layer
        x = self.gat3(x, edge_index)
        return x

import torch
import torch.nn.functional as F
import numpy as np

def contrastive_loss(embeddings, edge_index, margin=1.0, neg_pos_ratio=1):
    """
    Compute margin-based contrastive loss for graph embeddings.

    Args:
        embeddings (torch.Tensor): Node embeddings of shape (N, D).
        edge_index (torch.LongTensor): Edge indices of shape (2, E) for a directed graph.
        margin (float): Margin distance for negative pairs.
        neg_pos_ratio (int): Ratio of negatives to positives to sample.

    Returns:
        torch.Tensor: Scalar contrastive loss.
    """
    
    N = embeddings.size(0)

    # 1) Extract directed edges as pairs (u, v)
    directed_edges = set(tuple(edge.cpu().numpy()) for edge in edge_index.T)

    # 2) Symmetrize to get undirected positives
    positives = directed_edges | {(v, u) for (u, v) in directed_edges}
    num_pos = len(positives)

    # 3) Compute positive loss: pull connected nodes together
    pos_loss = torch.tensor(0.0)
    for (i, j) in positives:
        dist = torch.norm(embeddings[i] - embeddings[j], p=2)
        pos_loss += dist**2

    # 4) Identify all possible pairs and sample true negatives
    all_pairs = {(i, j) for i in range(N) for j in range(N) if i != j}
    negatives = list(all_pairs - positives)
    num_neg = min(len(negatives), num_pos * neg_pos_ratio)
    sampled_indices = np.random.choice(len(negatives), size=num_neg, replace=False)

    # 5) Compute negative loss: push unconneed nodes apart by at least margin
    neg_loss = torch.tensor(0.0)
    for idx in sampled_indices:
        i, j = negatives[idx]
        dist = torch.norm(embeddings[i] - embeddings[j], p=2)
        neg_loss += F.relu(margin - dist)**2

    # 6) Combine and normalize by total pairs
    loss = (pos_loss + neg_loss) / (num_pos + num_neg)
    return loss

# Reminder: In your GNN forward, normalize embeddings before computing loss:
# return F.normalize(x, p=2, dim=1)




# Get input dimension from the first graph
input_dim = train_graphs[0].x.shape[1]  # Number of node features


# Define learning rate
learning_rate = 0.001
# Number of epochs
epochs = 80


def train_and_validate(input_dim, hidden_dim, output_dim, num_heads, learning_rate, margin, train_graphs, val_graphs, epochs):
    model = GATModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for graph in train_graphs:
            optimizer.zero_grad()
            x, edge_index = graph.x, graph.edge_index
            embeddings = model(x, edge_index)
            loss = contrastive_loss(embeddings, edge_index, margin)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for graph in val_graphs:
                x, edge_index = graph.x, graph.edge_index
                embeddings = model(x, edge_index)
                val_loss = contrastive_loss(embeddings, edge_index, margin)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_graphs)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        scheduler.step()

    return best_val_loss

# Search loop
from itertools import product

search_results = []

for hd, od, m, nh in product(hidden_dims, output_dims, margins, num_heads):
    print(f"Trying: HD={hd}, OD={od}, Margin={m}, Heads={nh}")
    val_loss = train_and_validate(input_dim=train_graphs[0].x.shape[1],
                                  hidden_dim=hd,
                                  output_dim=od,
                                  num_heads=nh,
                                  learning_rate=learning_rate,
                                  margin=m,
                                  train_graphs=train_graphs,
                                  val_graphs=val_graphs,
                                  epochs= epochs)
    search_results.append(((hd, od, m, nh), val_loss))
# Find best hyperparameters
best_hyperparams = min(search_results, key=lambda x: x[1])
print(f"Best Hyperparameters: {best_hyperparams[0]}")
print(f"Best Validation Loss: {best_hyperparams[1]}")

# Unpack best hyperparameters
best_hd, best_od, best_margin, best_nh = best_hyperparams[0]

# Build the final best model
final_model = GATModel(input_dim=input_dim, hidden_dim=best_hd, output_dim=best_od, num_heads=best_nh)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)
final_scheduler = torch.optim.lr_scheduler.StepLR(final_optimizer, step_size=10, gamma=0.5)


# Train the final model with the best hyperparameters
for epoch in range(epochs):
    final_model.train()
    total_train_loss = 0
    for graph in train_graphs:
        final_optimizer.zero_grad()
        x, edge_index = graph.x, graph.edge_index
        embeddings = final_model(x, edge_index)
        loss = contrastive_loss(embeddings, edge_index, margin=best_margin)
        loss.backward()
        final_optimizer.step()
        total_train_loss += loss.item()

    final_scheduler.step()

# Save final model
save_model = os.path.join(model_dir, "best_hpo_gat_model.pth")
torch.save(final_model.state_dict(), save_model )
print("Best model saved after hyperparameter optimization.")



def test_model_and_extract_embeddings(model, test_graphs, out_name="node_embeddings.pth"):
    """
    Test the GAT model and extract node embeddings for clustering.

    Parameters:
    - model: The trained GAT model.
    - test_graphs: List of graphs for testing.
    - out_name: Path to save the node embeddings.

    Returns:
    - embeddings_dict: A dictionary of node embeddings for each test graph.
    """
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    embeddings_dict = {}  # Dictionary to store embeddings for each graph

    with torch.no_grad():  # Disable gradient computation
        for i, graph in enumerate(test_graphs):
            x, edge_index = graph.x, graph.edge_index
            
            # Forward pass to get node embeddings
            embeddings = model(x, edge_index)
            
            # Compute the test loss
            test_loss = contrastive_loss(embeddings, edge_index, margin = best_margin)
            total_test_loss += test_loss.item()
            
            # Save embeddings for the current graph
            embeddings_dict[f"graph_{i}"] = embeddings.cpu()  # Move to CPU
    # Average test loss
    avg_test_loss = total_test_loss / len(test_graphs)
    print(f"Test Loss: {avg_test_loss:.4f}")
    save_embeddings = os.path.join(result_dir, out_name)
    # Save node embeddings to a file
    torch.save(embeddings_dict, save_embeddings)
    print(f"Node embeddings saved to {save_embeddings}.")

    return embeddings_dict


# Load the best model for testing
model_path = os.path.join(model_dir, "best_hpo_gat_model.pth")
final_model.load_state_dict(torch.load(model_path))
print("Testing the best model...")

# Test the model and extract node embeddings
node_embeddings = test_model_and_extract_embeddings(final_model, test_graphs, out_name="node_embeddings.pth")

# Extract embeddings of test graph
node_embeddings = node_embeddings["graph_0"]
embeddings_np = node_embeddings.detach().cpu().numpy()
# Print the node embeddings
print("Node embeddings for the test graph:")
print(node_embeddings)
print("Shape of embeddings_np:", embeddings_np.shape) # Print the shape of the embeddings

# Check the average, min, and max norms of the embeddings, 
# Average norm ≈ 1	Embeddings are normalized (unit vectors).
# Average norm ≈ 2-5	Small to moderate scale.
# Average norm > 10	Large scale (embeddings very spread out).
norms = np.linalg.norm(embeddings_np, axis=1)  # L2 norm of each node's embedding
print(f"Average embedding norm: {np.mean(norms):.4f}")
print(f"Min embedding norm: {np.min(norms):.4f}")
print(f"Max embedding norm: {np.max(norms):.4f}")

avg_distance = np.mean(pdist(embeddings_np, metric='euclidean'))
print(f"Average pairwise Euclidean distance: {avg_distance:.4f}")

print(f"Embedding min value: {embeddings_np.min():.4f}")
print(f"Embedding max value: {embeddings_np.max():.4f}")


# Extract the edge index of the test graph
test_graph = test_graphs[0]  # Since there is only one graph in the test set
edge_index = test_graph.edge_index

# Print the edge index for debugging or understanding
print("Edge index of the test graph:")
print(edge_index)



# Compute Pairwise Euclidean Distance Matrix
distance_matrix = squareform(pdist(embeddings_np, metric='euclidean'))
# Plot Heatmap of Distance Matrix
fig = plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, cmap="Reds")  # Use "Reds" colormap
plt.title("Heatmap of Pairwise Euclidean Distances (Red)")
plt.xlabel("Node Index")
plt.ylabel("Node Index")
heatmap_path = os.path.join(fig_dir, "distance_matrix_heatmap.png")
fig.savefig(heatmap_path, bbox_inches='tight')  # Save the figure
plt.close(fig)

# Compute Average Distances

# Connected pairs (edges)
connected_pairs = []
for u, v in edge_index.T.numpy():
    connected_pairs.append((int(u), int(v)))

connected_distances = []

for u, v in connected_pairs:
    connected_distances.append(distance_matrix[u, v])

avg_connected_distance = np.mean(connected_distances)
print(f"Average Euclidean Distance (Connected Nodes): {avg_connected_distance:.4f}")

# Compute Average Distance of Unconnected Node Pairs

# All possible pairs (excluding self-loops)
num_nodes = embeddings_np.shape[0]
all_pairs = set((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j)

# Unconnected pairs = all pairs - connected pairs
unconnected = list(set(all_pairs) - set(connected_pairs))

# Randomly sample unconnected pairs equal to number of connected pairs
np.random.seed(42)  # for reproducibility
sampled_unconnected_pairs = np.random.choice(len(unconnected), size=len(connected_pairs), replace=False)
sampled_unconnected_distances = []

for idx in sampled_unconnected_pairs:
    u, v = unconnected[idx]
    sampled_unconnected_distances.append(distance_matrix[u, v])

avg_unconnected_distance = np.mean(sampled_unconnected_distances)
print(f"Average Euclidean Distance (Unconnected Nodes): {avg_unconnected_distance:.4f}")

# Method to visualize the embeddings in a 2-D plot

# Apply t-SNE for dimensionality reduction to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=5)  # Initialize t-SNE with 2D output
embeddings_2d = tsne.fit_transform(embeddings_np)  # Reduce embeddings to 2D

#  Visualize t-SNE results with nodes
fig = plt.figure(figsize=(12, 8))  # Set figure size
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=50, c='blue', edgecolors='k')  # Plot nodes

# Add node labels (IDs)
for i, (x, y) in enumerate(embeddings_2d):
    plt.text(x, y, str(i), fontsize=9, ha='right', va='bottom', color='red')  # Add node ID as text

# Draw directed edges with arrowheads
for u, v in edge_index.T.numpy():
    plt.annotate(
        '',  # No text
        xy=(embeddings_2d[v, 0], embeddings_2d[v, 1]),  # Receiver position
        xytext=(embeddings_2d[u, 0], embeddings_2d[u, 1]),  # Sender position
        arrowprops=dict(
            arrowstyle='->',  # Arrowhead style
            color='gray',     # Edge color
            alpha=0.7,        # Transparency
            lw=1.5            # Line width
        )
    )

# Add plot title and show
plt.title("t-SNE Visualization of Node Embeddings with Task Graph Edges")
plt.axis("off")  # Hide axes for cleaner visualization
scatter_fig_path = os.path.join(fig_dir, "tsne_visualization.png")
fig.savefig(scatter_fig_path, bbox_inches='tight')  # Save the figure
plt.close(fig)  # Close the figure
print("t-SNE visualization saved as 'tsne_visualization.png'.")

