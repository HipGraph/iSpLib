from GCN import GNN
# from GCN import iSpLibPlugin
import cProfile, pstats, io
import builtins

N_RUNS = 10     # How many times to repeat the experiment

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



def run_test(embedding_size):
    g = GNN(embedding_size)
    g.train_GCN()
    g.test_GCN()

import cProfile
import pandas as pd


def without_isplib(e):
    builtins.FUSEDMM = False
    # iSpLibPlugin.unpatch_pyg()
    with cProfile.Profile() as pr:
        for _ in range(N_RUNS):
            run_test(e)
        
        # print(pr.getstats())

        df = pd.DataFrame(
            pr.getstats(),
            columns=['func', 'ncalls', 'ccalls', 'tottime', 'cumtime', 'callers']
        )
    # "Total CPU Time: ", 
    total = df['cumtime'].max() / N_RUNS
    # kernel = df[df['func'].str.contains('sparse_mm', na=False)]['cumtime'].values[0] / N_RUNS
    kernel = df[df['func'].str.contains('isplib.spmm_sum', na=False)]['cumtime'].values[0] / N_RUNS

    
    # print("WITHOUT ISPLIB:")
    # print(f"\t{'Total CPU Time:':20}", f'{total:>.3f} seconds')
    # print(f"\t{'Total Kernel Time:':20}", f'{kernel:>.3f} seconds')

    return total, kernel

def using_isplib(e):
    builtins.FUSEDMM = True
    # iSpLibPlugin.patch_pyg()

    with cProfile.Profile() as pr:
        for _ in range(N_RUNS):
            run_test(e)

        df = pd.DataFrame(
            pr.getstats(),
            columns=['func', 'ncalls', 'ccalls', 'tottime', 'cumtime', 'callers']
        )
    # "Total CPU Time: ", 
    total = df['cumtime'].max() / N_RUNS
    kernel = df[df['func'].str.contains('fusedmm', na=False)]['cumtime'].values[0] / N_RUNS

    # print("USING ISPLIB:")
    # print(f"\t{'Total CPU Time:':20}", f'{total:>.3f} seconds')
    # print(f"\t{'Total Kernel Time:':20}", f'{kernel:>.3f} seconds')
    
    return total, kernel


# for e in [32, 64, 128, 256, 512]:
#     print(f'\n\n==[For embedding size {e}:]==')
#     a, b = without_isplib(e)
#     c, d = using_isplib(e)

#     print('===')
#     print(f"\t{'CPU Speedup:':20}", f'{a/c:>1.2f}x')
#     print(f"\t{'Kernel Speedup:':20}", f'{b/d:>1.2f}x')

data = {}
for e in [32, 64, 128, 256, 512]:
    # print(f'\n\n==[For embedding size {e}:]==')
    a, b = without_isplib(e)
    c, d = using_isplib(e)
    data[e] = {
        'CPU Time': {
            'Non-iSpLib': a,
            'iSpLib': c,
            'Speedup': a/c
        },
        'Kernel Time': {
            'Non-iSpLib': b,
            'iSpLib': d,
            'Speedup': b/d
        },
    }
    # print('===')
    # print(f"\t{'CPU Speedup:':20}", f'{a/c:>1.2f}x')
    # print(f"\t{'Kernel Speedup:':20}", f'{b/d:>1.2f}x')

data_df = pd.DataFrame.from_dict({(i, j): data[i][j] for i in data.keys() for j in data[i].keys()}, orient='index')
data_df.rename_axis(["Embedding_Size", 'Timing'], inplace=True)
print(data_df)
# print(pd.DataFrame.from_dict(data, orient='index'))



# cProfile.run('run_test()')



# df.to_excel('out.xlsx')
# with open('out.txt') as f:
#     for i in pr.getstats():
#         f.write(i)

# for i in pr.getstats():
#     print(i)

# >>> print(df[df['func'].str.contains('sparse_mm', na=False)])



# iSpLibPlugin.patch_pyg()
# cProfile.run('run_test()')



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
    