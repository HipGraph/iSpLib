from isplib import *
# Install PyTorch Geometric
import torch
# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Visualization
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 24})

from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T

dataset = TUDataset(root='./datasets', name='PROTEINS', transform=T.ToSparseTensor()).shuffle()

# Print information about the dataset
# print(f'Dataset: {dataset}')
# print('-------------------')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of nodes: {dataset[0].x.shape[0]}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')

# from torch_geometric.utils import to_networkx
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# G = to_networkx(dataset[2], to_undirected=True)

# # 3D spring layout
# pos = nx.spring_layout(G, dim=3, seed=0)

# # Extract node and edge positions from the layout
# node_xyz = np.array([pos[v] for v in sorted(G)])
# edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

# # Create the 3D figure
# fig = plt.figure(figsize=(16,16))
# ax = fig.add_subplot(111, projection="3d")

# # Suppress tick labels
# for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
#     dim.set_ticks([])

# # Plot the nodes - alpha is scaled by "depth" automatically
# ax.scatter(*node_xyz.T, s=500, c="#0A047A")

# # Plot the edges
# for vizedge in edge_xyz:
#     ax.plot(*vizedge.T, color="tab:gray")

# # fig.tight_layout()
# # plt.show()

from torch_geometric.loader import DataLoader

# Create training, validation, and test sets
train_dataset = dataset[:int(len(dataset)*0.8)]
val_dataset   = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
test_dataset  = dataset[int(len(dataset)*0.9):]

# print(f'Training set   = {len(train_dataset)} graphs')
# print(f'Validation set = {len(val_dataset)} graphs')
# print(f'Test set       = {len(test_dataset)} graphs')

# Create mini-batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# print('\nTrain loader:')
# for i, subgraph in enumerate(train_loader):
#     print(f' - Subgraph {i}: {subgraph}')

# print('\nValidation loader:')
# for i, subgraph in enumerate(val_loader):
#     print(f' - Subgraph {i}: {subgraph}')

# print('\nTest loader:')
# for i, subgraph in enumerate(test_loader):
#     print(f' - Subgraph {i}: {subgraph}')


import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h, F.log_softmax(h, dim=1)

# gcn = GCN(dim_h=32)
gin = GIN(dim_h=16)


def train(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=0.01,
                                      weight_decay=0.01)
    epochs = 100

    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Train on batches
        for data in loader:
          optimizer.zero_grad()
          _, out = model(data.x, data.adj_t, data.batch)
          loss = criterion(out, data.y)
          total_loss += loss / len(loader)
          acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
          loss.backward()
          optimizer.step()

          # Validation
          val_loss, val_acc = test(model, val_loader)

    # Print metrics every 10 epochs
    if(epoch % 10 == 0):
        print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
              f'| Train Acc: {acc*100:>5.2f}% '
              f'| Val Loss: {val_loss:.2f} '
              f'| Val Acc: {val_acc*100:.2f}%')
          
    test_loss, test_acc = test(model, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')
    
    return model

@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        _, out = model(data.x, data.adj_t, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

# print(matmul)
# iSpLibPlugin.patch_pyg()
# print(matmul)
# gcn = train(gcn, train_loader)

import cProfile

# cProfile.run('gin = train(gin, train_loader)')
gin = train(gin, train_loader)

gin.eval()

def test_GIN():
    acc_gin = 0
    acc = 0

    for data in test_loader:
        _, out_gin = gin(data.x, data.adj_t, data.batch)
        
        acc_gin += accuracy(out_gin.argmax(dim=1), data.y) / len(test_loader)
    return acc_gin


# print(f'GIN accuracy:     {acc_gin*100:.2f}%')

import cProfile, pstats
from pstats import SortKey

# https://gist.github.com/romuald/0346c76cfbbbceb3e4d1

def f8(x):
    ret = "%8.6f" % x
    if ret != '   0.000':
        return ret
    return "%6dµs" % (x * 1000000)

pstats.f8 = f8

import io

def get_cumulative_time(FusedMM):
    with cProfile.Profile() as pr:
        test_GIN()
        txt = io.StringIO()
        p = pstats.Stats(pr, stream=txt)
        p.print_stats('sparse.mm' if not FusedMM else 'isplib.fusedmm_spmm')
        # print(txt.getvalue())
        return txt.getvalue().strip().split('\n')[-1].split(' ')[-4]


torch_op_time = float(get_cumulative_time(False))

iSpLibPlugin.patch_pyg()
fusedmm_time = float(get_cumulative_time(True))
iSpLibPlugin.unpatch_pyg()

speedup = torch_op_time / fusedmm_time

print("Non-FusedMM SpMM time: ", torch_op_time, 'seconds')
print("FusedMM SpMM time: ", fusedmm_time, 'seconds')
print()
print("Speedup: ", f'{speedup:.3}x')
