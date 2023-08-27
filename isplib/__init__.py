import importlib
import os.path as osp

import torch
import torch_sparse
# print("OK")
from torch_sparse import SparseTensor, matmul
import torch_geometric.typing

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
    global matmul
    # try:
    # import torch_sparse
    try:
        def spmm_autotuned(src, other, reduce: str = "sum") -> torch.Tensor:
            # print(src.csr())
            # if not isinstance(src, SparseTensor):
            #     # print(src)
            #     src_backup = src
            #     src = torch_sparse.SparseTensor.from_torch_sparse_csr_tensor(src)
            rowptr, col, value = src.csr()
            if value is None:
               value = torch.ones_like(col, dtype=torch.float32)
            #    print('updated')
            # print('isplib', value is not None)
            # print('splib', value is None)

            row = src.storage._row
            rowcount = src.storage._rowcount
            # rowcount = None
            csr2csc = src.storage._csr2csc
            colptr = src.storage._colptr

            # if value is None:
            # #    src.storage._value = torch.ones_like(col)
            #    value = torch.ones_like(col)
            # #    print(src.csr())
            # #    print('updated')

            if value is not None:
                value = value.to(other.dtype)

            if value is not None and value.requires_grad:
                row = src.storage.row()

            if other.requires_grad:
                row = src.storage.row()
                rowcount = src.storage.rowcount()
                csr2csc = src.storage.csr2csc()
                colptr = src.storage.colptr()
            
            # for i in [row, rowptr, col, value, rowcount, colptr, csr2csc, other]:
            #       if i is None:
            #          print(None)
            #       else:
            #          print(i.dtype)
            # print()
            # print('Using FusedMM SpMM...')
            # print(other, reduce)
            # # Max
            # a = torch.ops.isplib.fusedmm_spmm_max(rowptr, col, value, other)
            # b = torch.ops.torch_sparse.spmm_max(rowptr, col, value, other)
            # # b = iSpLibPlugin.backup[1](src_backup, other, "max")

            # # Min
            # a = torch.ops.isplib.fusedmm_spmm_min(rowptr, col, value, other)
            # b = torch.ops.torch_sparse.spmm_min(rowptr, col, value, other)
            # # b = iSpLibPlugin.backup[1](src_backup, other, "min")

            # # Mean
            # a = torch.ops.isplib.fusedmm_spmm_mean(row, rowptr, col, value, rowcount,
            #                                 colptr, csr2csc, other)
            # b = torch.ops.torch_sparse.spmm_mean(row, rowptr, col, value, rowcount,
            #                                 colptr, csr2csc, other)
            # print(src, other.size(), reduce)
            # print('---')
            # print(a)
            # print(b)
            # print('---')
            if reduce in ['sum', 'add']:
               out = torch.ops.isplib.fusedmm_spmm(row, rowptr, col, value, colptr, csr2csc, other)
            elif reduce == 'max':
               out = torch.ops.isplib.fusedmm_spmm_max(rowptr, col, value, other)
            elif reduce == 'min':
               out = torch.ops.isplib.fusedmm_spmm_min(rowptr, col, value, other)
            elif reduce == 'mean':
            #    print(row.dtype, rowptr.dtype, col.dtype, value, rowcount, colptr, csr2csc, other.dtype)
               
            #    print('Torchsparse output: ')
            #    print(torch.ops.torch_sparse.spmm_mean(row, rowptr, col, value, rowcount, colptr, csr2csc, other))
               out = torch.ops.isplib.fusedmm_spmm_mean(row, rowptr, col, value, rowcount, colptr, csr2csc, other)
            #    out = torch.ops.torch_sparse.spmm_mean(row, rowptr, col, value, rowcount, colptr, csr2csc, other)
               
            else:
               return None
            
            return out
            
        iSpLibPlugin.backup.append(torch_sparse.matmul)
        iSpLibPlugin.backup.append(torch.sparse.mm)
        # iSpLibPlugin.backup.append(torch_geometric.typing.WITH_PT2)
        # iSpLibPlugin.backup.append(torch_sparse.matmul)
        torch_sparse.matmul = spmm_autotuned
        torch.sparse.mm = spmm_autotuned
        # torch_geometric.typing.WITH_PT2 = False
        # torch_sparse.matmul = spmm_autotuned

        # print('>> Autotuner activated')
    # print("Redirected")
    except Exception as e:
      print("Error! ", str(e))
  
  @classmethod
  def unpatch_pyg(self):
    global matmul
    if len(iSpLibPlugin.backup) > 0:
      # torch_sparse.matmul = iSpLibPlugin.backup.pop()
    #   iSpLibPlugin.backup.pop()
    #   torch_geometric.typing.WITH_PT2 = iSpLibPlugin.backup.pop()
      torch.sparse.mm = iSpLibPlugin.backup.pop()
      torch_sparse.matmul = iSpLibPlugin.backup.pop()
    #   print('<< Autotuner deactivated')


def isplib_autotune(fn):
    def wrapper(*args, **kwargs):    
        iSpLibPlugin.patch_pyg()
        ret = fn(*args, **kwargs)
        iSpLibPlugin.unpatch_pyg()
        return ret
    return wrapper