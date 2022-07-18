import torch
from the_sparse_package.tensor import SparseTensor


def random_walk(src: SparseTensor, start: torch.Tensor,
                walk_length: int) -> torch.Tensor:
    rowptr, col, _ = src.csr()
    return torch.ops.the_sparse_package.random_walk(rowptr, col, start, walk_length)


SparseTensor.random_walk = random_walk
