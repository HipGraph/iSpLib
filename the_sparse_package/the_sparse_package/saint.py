from typing import Tuple

import torch
from the_sparse_package.tensor import SparseTensor


def saint_subgraph(src: SparseTensor, node_idx: torch.Tensor
                   ) -> Tuple[SparseTensor, torch.Tensor]:
    row, col, value = src.coo()
    rowptr = src.storage.rowptr()

    data = torch.ops.the_sparse_package.saint_subgraph(node_idx, rowptr, row, col)
    row, col, edge_index = data

    if value is not None:
        value = value[edge_index]

    out = SparseTensor(row=row, rowptr=None, col=col, value=value,
                       sparse_sizes=(node_idx.size(0), node_idx.size(0)),
                       is_sorted=True)

    return out, edge_index


SparseTensor.saint_subgraph = saint_subgraph
