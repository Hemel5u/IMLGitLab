import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define the modified star graph
def create_modified_star_graph():
    # Node features (5 nodes, each with 1-dimensional feature)
    x = torch.tensor([[1], [0], [0], [0], [0]], dtype=torch.float)

    # Edge list (COO format)
    edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 3, 4],
                               [1, 2, 3, 4, 0, 2, 1, 1, 0]], dtype=torch.long)

    # Node labels (1 for center, 0 for others)
    y = torch.tensor([1, 0, 0, 0, 0], dtype=torch.long)

    # Masks for training and testing
    train_mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)  # Train on Node 0 and 1
    test_mask = torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool)   # Test on outer nodes

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    return data

# Visualize the graph
def visualize_modified_star_graph(data):
    G = nx.Graph()
    edges = data.edge_index.t().tolist()  # Convert edge_index to list of tuples
    G.add_edges_from(edges)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Spring layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800, font_size=10)
    plt.title("Modified Star Graph Visualization")
    plt.show()

# GCN Model Definition
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create and visualize the modified star graph
data = create_modified_star_graph()
visualize_modified_star_graph(data)

# Initialize the GCN model
model = GCN(input_dim=1, hidden_dim=4, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
def test():
    model.eval()
    out = model(data)
    pred = out[data.test_mask].max(1)[1]
    acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

# Lists to store metrics
losses = []
test_accuracies = []

# Train the GCN and collect metrics
for epoch in range(50):
    loss = train()
    acc = test()
    losses.append(loss)
    test_accuracies.append(acc)
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

# Plot Training Loss and Testing Accuracy
plt.figure(figsize=(12, 5))

# Training Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, 51), losses, label='Training Loss', color='blue', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid()
plt.legend()

# Testing Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, 51), test_accuracies, label='Testing Accuracy', color='green', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy Over Epochs')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

