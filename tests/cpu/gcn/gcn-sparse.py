import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics
import time
import numpy as np

import torch_geometric.typing
torch_geometric.typing.WITH_PT2 = False
torch_geometric.typing.WITH_PT20 = False

from isplib import * 
iSpLibPlugin.patch_pyg()

EPOCH_COUNT = 1
EMBEDDING_SIZE = 64

from torch_geometric.datasets import Planetoid, Amazon, TUDataset, Reddit
from torch_geometric.graphgym.loader import load_ogb
# from ogb.nodeproppred import PygNodePropPredDataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import numpy as np
import torch_geometric.transforms as T

device = torch.device('cpu')

# transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm(), T.ToSparseTensor()]) # poor accuracy for reddit
# dataset = Reddit(root='./datasets/Reddit', transform=transform) # poor accuracy for reddit
# dataset = Reddit(root='./datasets/Reddit')

# ogbn-proteins
# dataset = PygNodePropPredDataset(name='ogbn-products', root='./datasets/ogbn-products', transform=T.ToSparseTensor())


#Reddit and Cora:
# dataset = Planetoid("Planetoid", name="Cora", transform=T.ToSparseTensor())
dataset = Reddit(root='./datasets/Reddit', transform=T.ToSparseTensor())
data = dataset[0].to(device)

#OGB:
# dataset = load_ogb('ogbn-products', './datasets/ogbn-products')
# data = (T.ToSparseTensor()(dataset[0])).to(device)


# dataset[0].adj_t.storage._value = torch.ones(dataset[0].adj_t.storage._col.shape[0])


# from isplib.tensor import SparseTensor

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, EMBEDDING_SIZE, cached=True)
        self.conv2 = GCNConv(EMBEDDING_SIZE, dataset.num_classes, cached=True)

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj_t)

        return F.log_softmax(x, dim=1)


model = Net().to(device)



# adj_mat = data.adj_t
# # print(adj_mat.storage._csr2csc)

# adj_mat.csr()
# adj_mat.storage.row()
# adj_mat.storage.csr2csc()
# adj_mat.storage.colptr()
# data.adj_t = adj_mat
# # print(data.adj_t.storage.csr2csc())


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
  print('Testing Accuracy: {:.4f}, Time: {:.4f}'.format(acc, time.time() - t))
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

# print(f'Average training time: {(time.time() - t)/EPOCH_COUNT}')

def run_all():
  print('Embedding Size:', EMBEDDING_SIZE)
  train_GCN()   
  print(f'Training time avg: {np.mean(train_times)} std: {np.std(train_times)}')
  test_GCN()

# cProfile.run('run_all()', sort=SortKey.CUMULATIVE)

run_all()

# py-spy record --native -o out.svg -f speedscope -- python GIN.py

# py-spy record --native --rate 30 -o out.svg -f speedscope -- python GIN.py
# python parse_fusedmm.py out.txt