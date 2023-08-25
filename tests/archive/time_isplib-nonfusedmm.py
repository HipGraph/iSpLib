from GCN import GNN
from GCN import iSpLibPlugin
import cProfile, pstats, io
# import builtins
import torch_geometric.typing

N_RUNS = 1     # How many times to repeat the experiment

# from pstats import SortKey
# def f8(x):
#     # Source: https://gist.github.com/romuald/0346c76cfbbbceb3e4d1
#     ret = "%8.6f" % x
#     if ret != '   0.000':
#         return ret
#     return "%6dµs" % (x * 1000000)

# pstats.f8 = f8

# callcount: number of times called
# recallcount: number of times recursively called
# totaltime: total time spent in the function
# inlinetime: time spent in the function but not in subcalls respectively
# Source: https://zameermanji.com/blog/2012/6/30/undocumented-cprofile-features/


# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/spmm.py


def run_test(embedding_size):
    g = GNN(embedding_size)
    print(g.train_GCN())
    print(g.test_GCN())

torch_geometric.typing.WITH_PT2 = False
run_test(32)