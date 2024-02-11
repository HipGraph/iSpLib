import subprocess
import sys
from tqdm import tqdm
dataset = sys.argv[1]
# kernel = sys.argv[2]
kernel = 'spmm'


# ==========================================================================
# How to generate mtx file from PyTorch datasets?
# # !pip install fast_matrix_market


# import scipy.sparse as sparse
# import fast_matrix_market as fmm

# def dataset_to_mtx(dataset, filename):
#     coo = dataset.adj_t.to_torch_sparse_coo_tensor()
#     coo = coo.coalesce()
#     i = coo.indices()[0] 
#     j = coo.indices()[1]
#     v = coo.values()
#     shape = coo.size()
#     coo_sci = sparse.coo_matrix((v,(i,j)),shape=(shape[0], shape[1]))
#     fmm.mmwrite(filename,coo_sci)
# dataset_to_mtx(b, "ogbn_protein.mtx")
# ==========================================================================

# for k in tqdm(range(16, 513, 16)):
# for k in tqdm([64]):
# for dataset in ['amazon_products', 'ogbn_mag', 'ogbn_protein', 'reddit', 'ogbn_product', 'reddit2']:
print(f'==For dataset: {dataset}===')
out = []
for k in tqdm([16, 32, 64, 128, 256, 512, 1024]):
    a = f'../bin/xsOptFusedMMtime_{kernel}_pt'
    b = f"../dataset/{dataset}.mtx"
    output = subprocess.check_output([a, "-input", b, "-K", f"{k}", "skHd", "1"]).decode('utf-8')
    out += [output]
    # out += [(float(output.split(',')[-2]), k)]d
for i in out:
    print(i, end='')

print('======')
print('')
print('')


# python findbestk.py ogbn-prot spmm > log/out_ogbn-prot_spmm.txt
# ../bin/xsOptFusedMMtime_spmm_pt -input ../dataset/harvard.mtx -K 16
# ./bin/xsOptFusedMMtime_spmm_pt -input dataset/ogbn.mtx 