

# from networkx import project
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics
import time
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


import torch_geometric.typing
torch_geometric.typing.WITH_PT2 = False
torch_geometric.typing.WITH_PT20 = False

from isplib import * 
iSpLibPlugin.patch_pyg()


EPOCH_COUNT = 10
EMBEDDING_SIZE = 64
scaling_factor = 2

from torch_geometric.datasets import Planetoid, Amazon, TUDataset, Reddit
# from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
# dataset = Planetoid("Planetoid", name="Cora", transform=T.ToSparseTensor())
dataset = Reddit(root='./datasets/Reddit', transform=T.ToSparseTensor())
# dataset = Reddit(root='../../GCN-original/PyG/datasets/Reddit')
# transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm(), T.ToSparseTensor()]) # poor accuracy for reddit
# dataset = Reddit(root='./datasets/Reddit', transform=transform) # poor accuracy for reddit
# dataset = Reddit(root='./datasets/Reddit')


# dataset = PygNodePropPredDataset(name='ogbn-products', transform=transform)
# dataset[0].adj_t.storage._value = torch.ones(dataset[0].adj_t.storage._col.shape[0])

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
# from isplib.tensor import SparseTensor

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(EMBEDDING_SIZE*scaling_factor, EMBEDDING_SIZE, cached=True, aggr='sum')
        self.conv2 = SAGEConv(EMBEDDING_SIZE, dataset.num_classes, cached=True, aggr='sum')
        self.lin1 = nn.Linear(dataset.num_node_features, EMBEDDING_SIZE*scaling_factor)
        self.lin2 = nn.Linear(EMBEDDING_SIZE, dataset.num_classes)

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.lin1(x)
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj_t)

        return F.log_softmax(x, dim=1)




class Net0(torch.nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.conv1 = SAGEConv(dataset.num_node_features, EMBEDDING_SIZE, cached=True, aggr='sum')
        self.conv2 = SAGEConv(EMBEDDING_SIZE, dataset.num_classes, cached=True, aggr='sum')

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj_t)

        return F.log_softmax(x, dim=1)


device = torch.device('cpu')
model = Net().to(device)
data = dataset[0].to(device)

train_times = []
def train_GCN():
  global train_times
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

  model.train() 
  for epoch in range(EPOCH_COUNT):
      t = time.time()
      optimizer.zero_grad()
      out = model(data)
      loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()    
      _, pred = model(data).max(dim=1)
      correct = float (pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
      acc = correct / data.train_mask.sum().item()
      t1 = time.time() - t
      train_times += [t1]
      print('Epoch: %d, Accuracy: %.4f, Time: %.4f'%(epoch,acc, t1))

def test_GCN():
  # print(data.adj_t)
  t = time.time()
  _, pred = model(data).max(dim=1)
  correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
  acc = correct / data.test_mask.sum().item()
  print('Accuracy: {:.4f}, Time: {:.4f}'.format(acc, time.time() - t))
#   return acc


import cProfile
from pstats import SortKey

# def ten_times():
#     for _ in range(10):
#         test_GCN()
# cProfile.run('ten_times()', sort=SortKey.CUMULATIVE)
# cProfile.run('train_GCN()', sort=SortKey.CUMULATIVE)



# cProfile.run('[test_GCN() for _ in range(10)]', sort=SortKey.CUMULATIVE, filename='PT2.pstats')
# cProfile.run('[test_GCN() for _ in range(1)]', sort=SortKey.CUMULATIVE)
# t = time.time()
print('Embedding Size:', EMBEDDING_SIZE)
train_GCN()
# print(f'Average training time: {(time.time() - t)/EPOCH_COUNT}')

print(f'Training time avg: {np.mean(train_times)} std: {np.std(train_times)}')

test_GCN()


# py-spy top -- python GIN.py --native
# py-spy record --native -o out.svg -f speedscope -- python GIN.py