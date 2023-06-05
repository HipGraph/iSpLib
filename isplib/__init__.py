import importlib
import os.path as osp

import torch

__version__ = '0.1.0'

for library in [
        '_fusedmm'
]:
    # cuda_spec = importlib.machinery.PathFinder().find_spec(
    #     f'{library}_cuda', [osp.dirname(__file__)])
    cpu_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                          f"{osp.dirname(__file__)}")

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
