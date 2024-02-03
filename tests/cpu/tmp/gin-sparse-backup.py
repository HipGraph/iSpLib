import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import ToSparseTensor
import time

import torch_geometric.typing
torch_geometric.typing.WITH_PT2 = False
torch_geometric.typing.WITH_PT20 = False

from isplib import * 
iSpLibPlugin.patch_pyg()

EMBEDDING_SIZE = 32
EPOCH_COUNT = 10
scaling_factor = 2


dataset = Reddit(root='../datasets/Reddit', transform=ToSparseTensor())
data = dataset[0]  # Get the first graph object.
data = data.to('cpu')

class GINNet(torch.nn.Module):
    def __init__(self):
        super(GINNet, self).__init__()
        nn1 = Sequential(Linear(EMBEDDING_SIZE*scaling_factor, EMBEDDING_SIZE), ReLU(), Linear(EMBEDDING_SIZE, EMBEDDING_SIZE))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(EMBEDDING_SIZE)

        nn2 = Sequential(Linear(EMBEDDING_SIZE, EMBEDDING_SIZE), ReLU(), Linear(EMBEDDING_SIZE, EMBEDDING_SIZE))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(EMBEDDING_SIZE)

        self.fc1 = Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.fc2 = Linear(EMBEDDING_SIZE, dataset.num_classes)
        
        self.lin1 = torch.nn.Linear(dataset.num_features, EMBEDDING_SIZE*scaling_factor)

    def forward(self, x, adj_t):
        x = self.lin1(x)
        x = self.bn1(self.conv1(x, adj_t))
        x = self.bn2(self.conv2(x, adj_t))
        x = self.fc1(x).relu()
        return self.fc2(x)
    

class GINNet0(torch.nn.Module):
    def __init__(self):
        super(GINNet0, self).__init__()
        nn1 = Sequential(Linear(dataset.num_features, EMBEDDING_SIZE), ReLU(), Linear(EMBEDDING_SIZE, EMBEDDING_SIZE))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(EMBEDDING_SIZE)

        nn2 = Sequential(Linear(EMBEDDING_SIZE, EMBEDDING_SIZE), ReLU(), Linear(EMBEDDING_SIZE, EMBEDDING_SIZE))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(EMBEDDING_SIZE)

        self.fc1 = Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.fc2 = Linear(EMBEDDING_SIZE, dataset.num_classes)

    def forward(self, x, adj_t):
        x = self.bn1(self.conv1(x, adj_t))
        x = self.bn2(self.conv2(x, adj_t))
        x = self.fc1(x).relu()
        return self.fc2(x)
    
# def accuracy(output, target):
#     _, predictions = output.max(dim=1)
#     correct = (predictions == target).sum().item()
#     return correct / target.size(0)

def masked_accuracy(output, target, mask):
    _, predictions = output.max(dim=1)
    correct = (predictions[mask] == target[mask]).sum().item()
    return correct / mask.sum().item()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = GINNet0().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print('Embedding Size: ', EMBEDDING_SIZE)
# model.train()
model.train()
train_times = []
for epoch in range(EPOCH_COUNT):
    t = time.time()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    train_acc = masked_accuracy(out, data.y, data.train_mask)
    t1 = time.time() - t
    train_times += [t1]
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Time: {t1:.4f}')


model.eval()
with torch.no_grad():
    t = time.time()
    out = model(data.x, data.adj_t)
    test_acc = masked_accuracy(out, data.y, data.test_mask)
    t1 = time.time() - t
    print(f'Test Accuracy: {test_acc:.4f}, Time: {t1:.4f}')

import numpy as np
print(f'Training time avg: {np.mean(train_times)} std: {np.std(train_times)}')

# py-spy record --rate 30 --native -o gin.svg -f speedscope -- python gin-sparse.py