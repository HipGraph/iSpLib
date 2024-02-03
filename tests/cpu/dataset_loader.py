from torch_geometric.datasets import Reddit, AmazonProducts,Reddit2
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
from torch_scatter import scatter_add
import numpy as np

def load_reddit(path='../datasets', device='cpu', adj_t=True):
    if adj_t:
        dataset = Reddit(root=f'{path}/Reddit', transform=T.ToSparseTensor())
    else:
        dataset = Reddit(root=f'{path}/Reddit')
    data = dataset[0].to(device)
    dataset.num_node_features_ = dataset.num_node_features
    dataset.num_classes_ = dataset.num_classes
    return dataset, data


def load_reddit2(path='../datasets', device='cpu', adj_t=True):
    if adj_t:
        dataset = Reddit2(root=f'{path}/Reddit2', transform=T.ToSparseTensor())
    else:
        dataset = Reddit2(root=f'{path}/Reddit2')
    data = dataset[0].to(device)
    dataset.num_node_features_ = dataset.num_node_features
    dataset.num_classes_ = dataset.num_classes
    return dataset, data


def load_ogbn_mag(path='../datasets', device='cpu', adj_t=True):
    dataset = PygNodePropPredDataset(name='ogbn-mag', root=f'{path}/ogbn-mag')
    data = dataset[0].to(device)

    data.num_nodes = data.num_nodes_dict['paper']
    data.x = data.x_dict['paper']
    data.y =  torch.squeeze(data.y_dict['paper'])
    
    N = data.num_nodes
    
    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.train_mask[dataset.get_idx_split()['train']['paper']] = True

    data.test_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask[dataset.get_idx_split()['test']['paper']] = True

    data.val_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask[dataset.get_idx_split()['valid']['paper']] = True

    data.edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]

    if adj_t:
        data = T.ToSparseTensor()(data)

    dataset.num_node_features_ = data.x.shape[1]
    dataset.num_classes_ = dataset.num_classes

    return dataset, data


def load_amazon_products(path='../datasets', device='cpu', adj_t=True):
    if adj_t:
        dataset = AmazonProducts(root=f'{path}/AmazonProducts', transform=T.ToSparseTensor())
    else:
        dataset = AmazonProducts(root=f'{path}/AmazonProducts')
    data = dataset[0].to(device)

    # data.y = data.y.sum(axis=1)
    data.y = data.y[:,0]
    dataset.num_node_features_ = data.x.shape[1]
    # dataset.num_classes_ = dataset.num_classes
    # dataset.num_classes_ = len(set({int(i) for i in data.y}))
    dataset.num_classes_ = 2

    return dataset, data


def load_ogbn_product(path='../datasets', device='cpu', adj_t=True):
    if adj_t:
        dataset = PygNodePropPredDataset(name='ogbn-products', root=f'{path}/ogbn-products',transform=T.ToSparseTensor())
    else:
        dataset = PygNodePropPredDataset(name='ogbn-products', root=f'{path}/ogbn-products')

    data = dataset[0].to(device)
    
    data.y = torch.squeeze(data.y)
    
    N = data.num_nodes

    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.train_mask[dataset.get_idx_split()['train']] = True

    data.test_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask[dataset.get_idx_split()['test']] = True

    data.val_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask[dataset.get_idx_split()['valid']] = True

    
    dataset.num_node_features_ = data.x.shape[1]
    dataset.num_classes_ = dataset.num_classes
    return dataset, data


def load_ogbn_protein(path='../datasets', device='cpu', adj_t=True):
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root=f'{path}/ogbn-proteins')
    data = dataset[0].to(device)
    
    # data.y = data.y.sum(axis=1)
    data.y = data.y[:,0]

    N = data.num_nodes

    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.train_mask[dataset.get_idx_split()['train']] = True

    data.test_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask[dataset.get_idx_split()['test']] = True

    data.val_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask[dataset.get_idx_split()['valid']] = True

    _, indices = data.edge_index

    data.x = scatter_add(data.edge_attr, indices, dim=0, dim_size=N)
    
    dataset.num_node_features_ = data.x.shape[1]
    
    if adj_t:
        data = T.ToSparseTensor()(data)
        
    # dataset.num_classes_ = len(set({int(i) for i in data.y}))
    dataset.num_classes_ = 2
    return dataset, data


def get_padding_size(feature_size, block_size=16):
    if feature_size % 16 == 0:
        return 0
    else:
        return (feature_size // 16 + 1) * 16 - feature_size

def pad_features(a, b):
    row_size = b.x.shape[0]
    pad_amount = get_padding_size(b.x.shape[1])
    if pad_amount == 0:
        return a, b
    else:
        b.x = torch.tensor(np.concatenate((b.x, torch.tensor(np.zeros((row_size, pad_amount), dtype=b.x.numpy().dtype), device=b.x.device)), axis=1))
        a.num_node_features_ = b.x.shape[1]
        print('New feature length =', a.num_node_features_)
        return a, b
    
loader_dict = {
    'reddit': load_reddit,
    'reddit2': load_reddit2,
    'amazon': load_amazon_products,
    'protein': load_ogbn_protein,
    'product': load_ogbn_product,
    'mag': load_ogbn_mag,
}

# def load_dataset_mp(name, transform=None, path='../datasets'):
#     dataset = None
#     if name == 'reddit':
#         dataset = Reddit(root=f'{path}/Reddit', transform=transform)
#     elif name == 'reddit2':
#         dataset = Reddit2(root=f'{path}/Reddit2', transform=transform)
#     elif name == 'amazon':
#         dataset = AmazonProducts(root=f'{path}/AmazonProducts', transform=transform)
#     elif name == 'protein':

#     elif name == 'product':
#         dataset = PygNodePropPredDataset(name='ogbn-products', root=f'{path}/ogbn-products',transform=transform)
#     elif name == 'mag':
#         dataset = PygNodePropPredDataset(name='ogbn-mag', root=f'{path}/ogbn-mag', transform=transform)

# loader_dict_mp = {
#     'reddit': load_reddit,
#     'reddit2': load_reddit2,
#     'amazon': load_amazon_products,
#     'protein': load_ogbn_protein,
#     'product': load_ogbn_product,
#     'mag': load_ogbn_mag,
# }