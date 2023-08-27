from GCN import GNN
from GCN import iSpLibPlugin
import cProfile, pstats, io
# import builtins
import torch_geometric.typing
import cProfile
import pandas as pd
import datetime

import sys, os, subprocess
from tqdm import tqdm

pd.set_option('display.float_format', lambda x: '%.3f' % x)
torch_geometric.typing.WITH_PT2 = False

# module load gcc; module load anaconda; source activate py39
# python tests/time_isplib.py gcn cora 1 10
# python tests/time_isplib.py gcn reddit 5 100

# Params ------------------------------------------------------------------------

EMBEDDINGS = [16, 32, 64, 128, 256]
DEVICE = 'cpu'
WRITE_RESULTS_TO_FILE = True

GPROF_PATH = f'results/gprof2dot.py'
GENERATE_GRAPH = True
PYTHON = 'python'
# To visualize: python -m snakeviz prof.pstats

DEBUG = False

# ---

GNN_TYPE = sys.argv[1]
DATASET_NAME = sys.argv[2]
N_RUNS = int(sys.argv[3])     # How many times to repeat the experiment
EPOCH_COUNT = int(sys.argv[4])

# GNN_TYPE = 'gcn'
# DATASET_NAME = 'cora'
# N_RUNS = 1     # How many times to repeat the experiment
# EPOCH_COUNT = 10

NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
_EMBEDDINGS_STR = '-'.join(str(v) for v in EMBEDDINGS)
EXPERIMENT_NAME = f'{NOW}_{GNN_TYPE}-{DATASET_NAME}-EMB[{_EMBEDDINGS_STR}]N{N_RUNS}E{EPOCH_COUNT}_{DEVICE}'
OUTPUT_FOLDER = f'results/{EXPERIMENT_NAME}'

if GENERATE_GRAPH and not os.path.isfile(GPROF_PATH):
    print("Error: gprof2dot not found. Please put gprof2dot.py in ./results folder and run script again!\n---\nDownload: https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py \n---\n")
    GENERATE_GRAPH = False

if GENERATE_GRAPH:
    WRITE_RESULTS_TO_FILE = True

if WRITE_RESULTS_TO_FILE:
    os.makedirs(OUTPUT_FOLDER)


# -----------------------------------------------------------------------------------

print("Running experiment:", EXPERIMENT_NAME)
print(f"WRITE_RESULTS_TO_FILE={WRITE_RESULTS_TO_FILE}, GENERATE_GRAPH={GENERATE_GRAPH}, DEBUG={DEBUG}")
def run_test(embedding_size):
    g = GNN(embedding_size, GNN_TYPE, DATASET_NAME, EPOCH_COUNT, DEVICE)
    a = g.train_GCN()
    b = g.test_GCN()
    if DEBUG:
        print('Accuracy: ', a, b)


def without_isplib(e):
    # builtins.FUSEDMM = False
    iSpLibPlugin.unpatch_pyg()
    with cProfile.Profile() as pr:
        for _ in range(N_RUNS):
            run_test(e)
        
        # print(pr.getstats())

        df = pd.DataFrame(
            pr.getstats(),
            columns=['func', 'ncalls', 'ccalls', 'tottime', 'cumtime', 'callers']
        )
        if WRITE_RESULTS_TO_FILE:
            pr.dump_stats(f'{OUTPUT_FOLDER}/non-isplib-E{e}.pstats')
    if GENERATE_GRAPH:
        subprocess.Popen(f'{PYTHON} {GPROF_PATH} -f pstats {OUTPUT_FOLDER}/non-isplib-E{e}.pstats | dot -Tpng -o {OUTPUT_FOLDER}/non-isplib-E{e}.png', shell=True)

    # "Total CPU Time: ", 
    total = max(df['cumtime'].max(), df['tottime'].max()) / N_RUNS
    # kernel = df[df['func'].str.contains('sparse_mm', na=False)]['cumtime'].values[0] / N_RUNS
    kernel = df[df['func'].str.contains('torch_sparse.spmm', na=False)]['cumtime'].values[0] / N_RUNS
    
    df.sort_values(['cumtime', 'tottime', 'ncalls'], ascending=False, inplace=True)
    if WRITE_RESULTS_TO_FILE:
        df.to_html(f'{OUTPUT_FOLDER}/non-isplib-E{e}.html')
    # print(df)
    if DEBUG:
        print("WITHOUT ISPLIB:")
        print(f"\t{'Total CPU Time:':20}", f'{total:>.3f} seconds')
        print(f"\t{'Total Kernel Time:':20}", f'{kernel:>.3f} seconds')

    return total, kernel

def using_isplib(e):
    # builtins.FUSEDMM = True
    iSpLibPlugin.patch_pyg()

    with cProfile.Profile() as pr:
        for _ in range(N_RUNS):
            run_test(e)

        df = pd.DataFrame(
            pr.getstats(),
            columns=['func', 'ncalls', 'ccalls', 'tottime', 'cumtime', 'callers']
        )
        if WRITE_RESULTS_TO_FILE:
            pr.dump_stats(f'{OUTPUT_FOLDER}/isplib-E{e}.pstats')
    
    if GENERATE_GRAPH:
        subprocess.Popen(f'{PYTHON} {GPROF_PATH} -f pstats {OUTPUT_FOLDER}/isplib-E{e}.pstats | dot -Tpng -o {OUTPUT_FOLDER}/isplib-E{e}.png', shell=True)

    # "Total CPU Time: ", 
    # print(df)
    total = max(df['cumtime'].max(), df['tottime'].max()) / N_RUNS
    kernel = df[df['func'].str.contains('fusedmm', na=False)]['cumtime'].values[0] / N_RUNS
    df.sort_values(['cumtime', 'tottime', 'ncalls'], ascending=False, inplace=True)
    # df.to_html('out-splib-gcn.html')
    if WRITE_RESULTS_TO_FILE:
        df.to_html(f'{OUTPUT_FOLDER}/isplib-E{e}.html')
    if DEBUG: 
        print("USING ISPLIB:")
        print(f"\t{'Total CPU Time:':20}", f'{total:>.3f} seconds')
        print(f"\t{'Total Kernel Time:':20}", f'{kernel:>.3f} seconds')    
    return total, kernel



data = {}
for e in tqdm(EMBEDDINGS):
    # print(f'\n\n==[For embedding size {e}:]==')
    c, d = using_isplib(e)
    a, b = without_isplib(e)
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

data_df = data_df.swaplevel()
data_df = data_df.groupby("Timing", as_index=False).apply(lambda x: x)

print("\n Experiment Name:", EXPERIMENT_NAME, '\n')
print(data_df)

if WRITE_RESULTS_TO_FILE:
    data_df.to_csv(f'{OUTPUT_FOLDER}/summary.csv')
    print(f'\nResults dumped in folder: ./{OUTPUT_FOLDER}')
