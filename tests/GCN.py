# from isplib import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics

from torch_geometric.datasets import Planetoid, Amazon
import torch_geometric.transforms as T
dataset = Planetoid("datasets/Planetoid", name="Cora", transform=T.ToSparseTensor())
# dataset = Amazon("datasets/Amazon",name='computers' ,transform=T.Compose([T.ToSparseTensor(),T.RandomNodeSplit()]))

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_sparse.tensor import SparseTensor

class Net(torch.nn.Module):
    def __init__(self, embedding_size=16):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, embedding_size, cached=True)
        self.conv2 = GCNConv(embedding_size, dataset.num_classes, cached=True)

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj_t)

        return F.log_softmax(x, dim=1)


class GNN:
    def __init__(self, emb_size) -> None:
        self.device = torch.device('cpu')
        self.model = Net(emb_size).to(self.device)
        self.data = dataset[0].to(self.device)

   
    def train_GCN(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.model.train()

        for epoch in range(100):
            optimizer.zero_grad()
            out = self.model(self.data)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            
            _, pred = self.model(self.data).max(dim=1)
            correct = float (pred[self.data.train_mask].eq(self.data.y[self.data.train_mask]).sum().item())
            acc = correct / self.data.train_mask.sum().item()
            # print('Epoch: %d, Accuracy: %.4f'%(epoch,acc))

    # def test_GCN(FusedMM):
    def test_GCN(self):
        _, pred = self.model(self.data).max(dim=1)
        correct = float (pred[self.data.test_mask].eq(self.data.y[self.data.test_mask]).sum().item())
        acc = correct / self.data.test_mask.sum().item()
        # print('Accuracy: {:.4f}'.format(acc))
        return acc



# import cProfile, pstats
# from pstats import SortKey

# # https://gist.github.com/romuald/0346c76cfbbbceb3e4d1

# def f8(x):
#     ret = "%8.6f" % x
#     if ret != '   0.000':
#         return ret
#     return "%6dµs" % (x * 1000000)

# pstats.f8 = f8

# # iSpLibPlugin.patch_pyg()
# print("## Training GCN...")
# # cProfile.run('train_GCN()')
# train_GCN()
# print("Done!")

# print("## Testing GCN...")
# print("Accuracy without FusedMM: ", test_GCN())

# iSpLibPlugin.patch_pyg()
# print("Accuracy with FusedMM: ", test_GCN())
# iSpLibPlugin.unpatch_pyg()

# import io

# def get_cumulative_time(FusedMM):
#     with cProfile.Profile() as pr:
#         test_GCN()
#         txt = io.StringIO()
#         p = pstats.Stats(pr, stream=txt)
#         p.print_stats('sparse.mm' if not FusedMM else 'isplib.fusedmm_spmm')
#         # print(txt.getvalue())
#         return txt.getvalue().strip().split('\n')[-1].split(' ')[-4]

# from tqdm import tqdm


# a = []
# b = []
# c = []
# print('TorchOp', 'FusedMM', 'Speedup', sep='\t')
# for i in range(1000):
#     torch_op_time = float(get_cumulative_time(False))

#     iSpLibPlugin.patch_pyg()
#     fusedmm_time = float(get_cumulative_time(True))
#     iSpLibPlugin.unpatch_pyg()


#     speedup = torch_op_time / fusedmm_time

#     print(f'{torch_op_time:3}', f'{fusedmm_time:.3}', f'{speedup:.3}', sep='\t')
    


# torch.ops.isplib.performDummySpMM(3)

# torch_op_time = float(get_cumulative_time(False))

# iSpLibPlugin.patch_pyg()
# fusedmm_time = float(get_cumulative_time(True))
# iSpLibPlugin.unpatch_pyg()

# speedup = torch_op_time / fusedmm_time

# print("Non-FusedMM SpMM time: ", torch_op_time, 'seconds')
# print("FusedMM SpMM time: ", fusedmm_time, 'seconds')
# print()
# print("Speedup: ", f'{speedup:.3}x')
# # torch.ops.isplib.performDummySpMM(3)