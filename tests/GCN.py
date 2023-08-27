from isplib import *

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
import sklearn.metrics as metrics

from torch_geometric.datasets import Planetoid, Amazon, TUDataset, Reddit
import torch_geometric.transforms as T


import inspect
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
# from torch_sparse.tensor import SparseTensor


def get_dataset(dataset_name):
    if dataset_name == 'cora':
        return Planetoid("datasets/Planetoid", name="Cora", transform=T.ToSparseTensor())
    elif dataset_name == 'reddit':
        return Reddit(root='./datasets', transform=T.ToSparseTensor()).shuffle()
    
    # dataset = Amazon("datasets/Amazon",name='computers' ,transform=T.Compose([T.ToSparseTensor(),T.RandomNodeSplit()]))

class Net(torch.nn.Module):
    def __init__(self, embedding_size=16, gnn_type='gcn', dataset=None):
        super(Net, self).__init__()
        if dataset is None:
            raise Exception("Dataset cannot be None")
        if gnn_type == 'gcn':
            self.conv1 = GCNConv(dataset.num_node_features, embedding_size, cached=True)
            self.conv2 = GCNConv(embedding_size, dataset.num_classes, cached=True)
        elif gnn_type == 'sage':
            self.conv1 = SAGEConv(dataset.num_node_features, embedding_size, cached=True)
            self.conv2 = SAGEConv(embedding_size, dataset.num_classes, cached=True)
        # elif gnn_type == 'gin':
        #     self.conv1 = GINConv(dataset.num_node_features, embedding_size, cached=True)
        #     self.conv2 = GINConv(embedding_size, dataset.num_classes, cached=True)
        # elif gnn_type == 'gat':
        #     self.conv1 = GATConv(dataset.num_node_features, embedding_size, cached=True)
        #     self.conv2 = GATConv(embedding_size, dataset.num_classes, cached=True)

    def forward(self, data):
        # print(data.adj_t)
        x, adj_t = data.x, data.adj_t
        # print(adj_t)
        # print('forward', adj_t.csr()[2] is None)
        # print(inspect.getsource(self.conv1.message_and_aggregate))
        # print(inspect.getsourcefile(self.conv1.message_and_aggregate))
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj_t)

        return F.log_softmax(x, dim=1)


class GNN:
    def __init__(self, emb_size, gnn_type='gcn', dataset_name='cora', epoch_count = 10, device='cpu') -> None:
        dataset = get_dataset(dataset_name)
        self.device = torch.device(device)
        self.model = Net(emb_size, gnn_type, dataset).to(self.device)
        self.data = dataset[0].to(self.device)
        self.epoch_count = epoch_count
        # print(self.data.adj_t.csr()[2])
        if self.data.adj_t.csr()[2] is None:
            self.data.adj_t.storage.set_value_(torch.ones_like(self.data.adj_t.storage.col(), dtype=torch.float32), 'csr')
            # print('updated')
        # print("GNN init", self.data.adj_t.csr()[2] is None)
   
    def train_GCN(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.model.train()

        for epoch in range(self.epoch_count):
            optimizer.zero_grad()
            out = self.model(self.data)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            
            _, pred = self.model(self.data).max(dim=1)
            correct = float (pred[self.data.train_mask].eq(self.data.y[self.data.train_mask]).sum().item())
            acc = correct / self.data.train_mask.sum().item()
            # print('Epoch: %d, Accuracy: %.4f'%(epoch,acc))
        _, pred = self.model(self.data).max(dim=1)
        correct = float (pred[self.data.train_mask].eq(self.data.y[self.data.train_mask]).sum().item())
        acc = correct / self.data.train_mask.sum().item()
        # print('Accuracy: {:.4f}'.format(acc))
        return acc

    # def test_GCN(FusedMM):
    def test_GCN(self):
        _, pred = self.model(self.data).max(dim=1)
        correct = float (pred[self.data.test_mask].eq(self.data.y[self.data.test_mask]).sum().item())
        acc = correct / self.data.test_mask.sum().item()
        # print('Accuracy: {:.4f}'.format(acc))
        return acc

