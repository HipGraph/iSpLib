# python cogdl-gcn-sparse.py reddit isplib

EPOCH_COUNT = 100
EMBEDDING_SIZE = 32

PRINT_TABLE = False

from tqdm import tqdm
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
import torch_geometric

import sys
from dataset_loader import loader_dict

if len(sys.argv) < 2:
  print('Required 2 arguments: dataset and mode (pt2, pt1, or isplib)')
  exit()

# if sys.argv[2] in ['pt1', 'isplib']:
#   import torch_geometric.typing
#   torch_geometric.typing.WITH_PT2 = False
#   torch_geometric.typing.WITH_PT20 = False

# if sys.argv[2] == 'isplib':
#   from isplib import * 
#   iSpLibPlugin.patch_pyg()


print(f'Running GCN, Epoch: {EPOCH_COUNT}, Embedding: {EMBEDDING_SIZE}, Dataset: {sys.argv[1]}, Mode: CogDL')

from ogb.nodeproppred import PygNodePropPredDataset
import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T

from cogdl.layers import GCNLayer
from cogdl.data import Graph

device = torch.device('cpu')

def to_cogdl_graph(data):
  cogdlgraph = Graph(x=data.x, edge_index=data.edge_index, y=data.y)
  cogdlgraph.train_mask = data.train_mask
  cogdlgraph.val_mask = data.val_mask
  cogdlgraph.test_mask = data.test_mask
  return cogdlgraph
  
dataset, data = loader_dict[sys.argv[1]](adj_t=False)
# CogDL Graph Obj
g = to_cogdl_graph(data)

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = GCNLayer(dataset.num_node_features_, EMBEDDING_SIZE, cached=True)
    self.conv2 = GCNLayer(EMBEDDING_SIZE, dataset.num_classes_, cached=True)
  
  def forward(self, cogdlgraph, x):
    x = self.conv1(cogdlgraph, x)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.conv2(cogdlgraph, x)
    return F.log_softmax(x, dim=1)


model = Net().to(device)
# if sys.argv[2] == 'pt2+compile':
#   model = torch_geometric.compile(model, dynamic=False, fullgraph=True)


train_times = []
def train_GCN():
  global train_times
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

  model.train() 
  for epoch in tqdm(range(EPOCH_COUNT)):
    t = time.time()
    optimizer.zero_grad()
    out = model(g, data.x)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()    
    _, pred = model(g, data.x).max(dim=1)
    correct = float (pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / data.train_mask.sum().item()
    t1 = time.time() - t
    train_times += [t1]
    if PRINT_TABLE:
      print('Epoch: %d, Accuracy: %.4f, Time: %.4f'%(epoch,acc, t1))
  return acc

def test_GCN():
  # print(data.adj_t)
  model.eval()
  with torch.no_grad():
    t = time.time()
    _, pred = model(g, data.x).max(dim=1)
    correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    # print('Testing Accuracy: {:.4f}, Time: {:.4f}'.format(acc, time.time() - t))
    return acc, time.time() - t


def run_all():
  # print('Embedding Size:', EMBEDDING_SIZE)
  d = train_GCN()
  a = np.mean(train_times)
  b = np.std(train_times)

  # print(f'Training time avg: {np.mean(train_times)} std: {np.std(train_times)}')
  e, c = test_GCN()
  print()
  print('TRG_TM','TRG_STD','TST_TM','TRG_ACC','TST_ACC', sep='\t')
  print('-'*40)
  
  print(f'{a:.4}\t{b:.4}\t{c:.4}\t{d:.4}\t{e:.4}')
  print()
  print(f'{a:.4},{b:.4},{c:.4},{d:.4},{e:.4}')
  print('---')
  print()

run_all()