# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gcn.py

import os.path as osp
import time
import sys
import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn import GCNConv

import torch_geometric.typing
torch_geometric.typing.WITH_PT2 = False
torch_geometric.typing.WITH_PT20 = False


# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:

device = torch.device('cpu')

# path = osp.dirname(osp.realpath(__file__))
# path = osp.join(path, '..', '..', 'data', 'Planetoid')
# dataset = Planetoid(
#     path, name='Cora', transform=T.Compose([
#         T.NormalizeFeatures(),
#         T.GCNNorm(),
#     ]))

# dataset = Reddit(root='./datasets/Reddit', transform=T.Compose([
#         T.NormalizeFeatures(),
#         T.GCNNorm(),
#     ]))

if sys.argv[1] == 'reddit':
    dataset = Reddit(root='../datasets/Reddit')
# elif sys.argv[1] == 'reddit2':
#     dataset = Reddit(root='../datasets/Reddit2')
# elif sys.argv[1] == 'mag':
#     dataset = Reddit(root='../datasets/ogbn-mag')
# elif sys.argv[1] == 'product':
#     dataset = Reddit(root='../datasets/ogbn-products')
# elif sys.argv[1] == 'protein':
#     dataset = Reddit(root='../datasets/ogbn-proteins')
# elif sys.argv[1] == 'amazon':
#     dataset = Reddit(root='../datasets/AmazonProducts')




data = dataset[0].to(device)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Pre-process normalization to avoid CPU communication/graph breaks:
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=32,
    out_channels=dataset.num_classes,
).to(device)

# Compile the model into an optimized version and enforce zero graph breaks:
model = torch_geometric.compile(model, dynamic=False, fullgraph=True)

optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


# times = []
for epoch in range(1, 10):
    start = time.time()
    loss = train()
    train_acc, val_acc, test_acc = test()
    # times.append(time.time() - start)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}', f'Time: {time.time() - start}')
# print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

t = time.time()
test()
print('Test time: ', time.time() - t)