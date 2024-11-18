# Required imports
import torch  # For building neural networks
import torch.nn.functional as F  # For activation functions like ReLU
from torch_geometric.nn import GATConv  # GAT layer from PyTorch Geometric
from sklearn.cluster import KMeans  # K-Means for clustering the embeddings


import json

# Load JSON data
with open(r'C:\Users\hites\Desktop\PROJECT ARBEIT\CODE\example_N1.json', "r") as file:
    data = json.load(file)

# Print structure
print(json.dumps(data, indent=4))

import torch

# Extract job information for node features
jobs = data['application']['jobs']
node_features = []
for job in jobs:
    features = [
        job['wcet_fullspeed'],
        job['processing_times'],
        job['deadline']
    ]
    node_features.append(features)

# Convert node features to a tensor
node_features = torch.tensor(node_features, dtype=torch.float)
print("Node Features:\n", node_features)

# Extract edges from messages
messages = data['application']['messages']
edges = []
for message in messages:
    edges.append([message['sender'], message['receiver']])
print("Edges", edges)

# Convert edge list to a tensor and transpose it for PyTorch Geometric
edge_index = torch.tensor(edges, dtype=torch.long).T
print("Edge Index:\n", edge_index)

# Define the GAT Model
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)
    
    def forward(self, x, edge_index):
        #First GAT Layer
        x = self.gat1(x, edge_index)  # Hidden Embeddings
        print("Hidden Embeddings:\n", x) #Print hidden embeddings
        x = F.relu(x)  # Non-linear activation
        # Second GAT LAyer
        x = self.gat2(x, edge_index)  # Output embeddings
        print("Output Embeddings:\n", x) # Print final output embeddings
        return x

# Initialize the GAT model
input_dim = node_features.shape[1]  # Number of input features per node
hidden_dim = 8  # Size of hidden embeddings
output_dim = 4  # Size of final node embeddings
num_heads = 4  # Number of attention heads

gat_model = GATModel(input_dim, hidden_dim, output_dim, num_heads)

# Pass the node features and edge index through the GAT model
node_embeddings = gat_model(node_features, edge_index)

print("Node Embeddings:\n", node_embeddings)
