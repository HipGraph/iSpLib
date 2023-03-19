import builtins
builtins.FUSEDMM = True


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
dataset = Planetoid("Planetoid", name="Cora", transform=T.ToSparseTensor())

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from isplib.tensor import SparseTensor

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj_t)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)

def train_GCN():
  builtins.FUSEDMM = True
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

  model.train()
  for epoch in range(10):
      optimizer.zero_grad()
      out = model(data)
      loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      
      _, pred = model(data).max(dim=1)
      correct = float (pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
      acc = correct / data.train_mask.sum().item()
      print('Epoch: %d, Accuracy: %.4f'%(epoch,acc))

def test_GCN(status):
  builtins.FUSEDMM = status  # Use FusedMM or not
  _, pred = model(data).max(dim=1)
  correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
  acc = correct / data.test_mask.sum().item()
  # print('Accuracy: {:.4f}'.format(acc))
  return acc

import cProfile, pstats
from pstats import SortKey

# https://gist.github.com/romuald/0346c76cfbbbceb3e4d1

def f8(x):
    ret = "%8.6f" % x
    if ret != '   0.000':
        return ret
    return "%6dµs" % (x * 1000000)

pstats.f8 = f8

print("Training GCN...")
train_GCN()
print("Done!")

print("Accuracy without FusedMM: ", test_GCN(False))
print("Accuracy with FusedMM: ", test_GCN(True))

import io

def get_cumulative_time(FusedMM=False):
    with cProfile.Profile() as pr:
        test_GCN(FusedMM)
        txt = io.StringIO()
        p = pstats.Stats(pr, stream=txt)
        p.print_stats('isplib.spmm_sum' if not FusedMM else 'isplib.fusedmm_spmm')
        # print(txt.getvalue())
        return txt.getvalue().strip().split('\n')[-1].split(' ')[-4]
    
torch_op_time = float(get_cumulative_time(False))
fusedmm_time = float(get_cumulative_time(True))
speedup = torch_op_time / fusedmm_time

print("Non-FusedMM SpMM time: ", torch_op_time, 'seconds')
print("FusedMM SpMM time: ", fusedmm_time, 'seconds')
print()
print("Speedup: ", f'{speedup:.3}x')