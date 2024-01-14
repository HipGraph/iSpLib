#include "generated/include/spmm_header.cuh"

__global__ void fusedmm_kernel(int m, int n, int k, int nnz, 
                               const int64_t* indx, const int64_t* ptrb, const float* val, 
                               const float* b, float* c)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    
    for (int i = threadId; i < m; i += num_threads) {
        int start_idx = ptrb[i];
        int end_idx = ptrb[i + 1];

        for (int j = start_idx; j < end_idx; ++j) {
            int colIdx = indx[j];
            float valElem = val[j];

            for (int colId = 0; colId < k; ++colId) {
                c[i * k + colId] += valElem * b[colIdx * k + colId];
            }
        }
    }
}

void fusedmm_spmm_trusted_kernel(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c) 
{   
    int threads_per_block = 256;
    int num_blocks = (m + threads_per_block - 1) / threads_per_block;
    fusedmm_kernel<<<num_blocks, threads_per_block>>>(m, n, k, nnz, indx, ptrb,val,b,c);
}



