#include "generated/include/spmm_header.cuh"

__global__ void fusedmm_spmm_trusted_ckernel(int m, int n, int k, int nnz, const int64_t* indx, const int64_t* ptrb, const float* val, const float* b, float* c, int nnz_per_block)
{
    int nnz_start = blockIdx.x * nnz_per_block;
    int nnz_end = min(nnz_start + nnz_per_block, nnz);
    
    for (int i = nnz_start; i < nnz_end; i++) {
        
	int row = ptrb[i];
        int col = indx[i];
        float val_ij = val[i];
	
        float temp = 0.0f;
	for (int j = threadIdx.x; j < k; j += blockDim.x) {
	    temp += val_ij * b[col * k + j];
        }
        atomicAdd(&c[row * k + threadIdx.x], temp);
    }
}

void fusedmm_spmm_trusted_kernel(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c) {

	int nnz_per_block = 128;
    int threads_per_block = 256;
	int num_blocks = (nnz + nnz_per_block - 1) / nnz_per_block;
    	
	//printf("Num Blocks: %d\n", num_blocks);
	
	fusedmm_spmm_trusted_ckernel<<<num_blocks, threads_per_block>>>(m, n, k, nnz, indx, ptrb, val, b, c, nnz_per_block);

	cudaDeviceSynchronize();
}
