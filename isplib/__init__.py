import importlib
import os.path as osp

import torch
import torch_sparse
from torch_sparse import SparseTensor

__version__ = '0.1.0'

for library in [
        '_fusedmm'
]:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cuda', [osp.dirname(__file__)])
    cpu_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                          f"{osp.dirname(__file__)}")



#init.py

class iSpLibPlugin:
  backup = []    

  @classmethod
  def patch_pyg(self):
    # try:
    # import torch_sparse
    

    def spmm_autotuned(src, other, reduce: str = "sum") -> torch.Tensor:
        if not isinstance(src, SparseTensor):
            # print(src)
            src = torch_sparse.SparseTensor.from_torch_sparse_csr_tensor(src)
        rowptr, col, value = src.csr()

        row = src.storage._row
        csr2csc = src.storage._csr2csc
        colptr = src.storage._colptr

        if value is not None:
            value = value.to(other.dtype)

        if value is not None and value.requires_grad:
            row = src.storage.row()

        if other.requires_grad:
            row = src.storage.row()
            csr2csc = src.storage.csr2csc()
            colptr = src.storage.colptr()

        print('Using FusedMM SpMM...')
        return torch.ops.isplib.fusedmm_spmm(row, rowptr, col, value, colptr, csr2csc, other)
        
    iSpLibPlugin.backup.append(torch_sparse.spmm)
    iSpLibPlugin.backup.append(torch.sparse.mm)
    torch_sparse.spmm = spmm_autotuned
    torch.sparse.mm = spmm_autotuned
    print("Redirected")
    # except Exception as e:
    #   print("Error! ", str(e))
  
  @classmethod
  def unpatch_pyg(self):
    if len(iSpLibPlugin.backup) > 0:
      torch.sparse.mm = iSpLibPlugin.backup.pop()
      torch_sparse.spmm = iSpLibPlugin.backup.pop()
      print("Restored!")

# cuda_version = torch.ops.isplib.cuda_version()
# if torch.version.cuda is not None and cuda_version != -1:  # pragma: no cover
#     if cuda_version < 10000:
#         major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
#     else:
#         major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
#     t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

#     if t_major != major:
#         raise RuntimeError(
#             f'Detected that PyTorch and isplib were compiled with '
#             f'different CUDA versions. PyTorch has CUDA version '
#             f'{t_major}.{t_minor} and isplib has CUDA version '
#             f'{major}.{minor}. Please reinstall the isplib that '
#             f'matches your PyTorch install.')

# from .storage import SparseStorage  # noqa
# from .tensor import SparseTensor  # noqa
# from .diag import fill_diag
# from .matmul import matmul  # noqa
# from .mul import mul
# from .reduce import sum

# __all__ = [
#     'SparseStorage',
#     'SparseTensor',
#     'matmul',
#     'sum',
#     'mul',
#     'fill_diag',
#     '__version__',
# ]
