#include "generated/include/spmm_header.cuh"

/*
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
}*/
/*
__global__ void fusedmm_kernel(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c)
{
        int rowId = blockIdx.x; // Sparse Matrix
        // int colId  = threadIdx.x; // Dense Matrix
        int start_idx = ptrb[rowId];
        int end_idx = ptrb[rowId + 1];

	__shared__ float sharedVal[256];
        __shared__ float sharedCol[256];

        if(threadIdx.x == 0){
                int len = end_idx - start_idx;
                int i = 0;
                while(i<len){
                        sharedVal[i] = val[start_idx+i];
                        sharedCol[i] = indx[start_idx+i];
                        i++;
                }
        }
        __syncthreads();


        for(int colId = threadIdx.x; colId < k; colId += blockDim.x){

                float buff = 0.0f;
                for(int i=start_idx; i<end_idx; i++){
                        int offset = sharedCol[i-start_idx]*k + colId;
                        buff += sharedVal[i-start_idx] * b[offset];
                }
                c[rowId*k + colId] = buff;
        }
}

void fusedmm_spmm_trusted_kernel(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c)
{
        int num_blocks = m;
        int threads_per_block = 16;
        fusedmm_kernel<<<num_blocks, threads_per_block>>>(m, n, k, nnz, indx, ptrb, val, b, c);
}
*/
// 0.7 Speedup version
/*
__global__ void fusedmm_kernel(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c)
{
        int rowId = blockIdx.x; // Sparse Matrix
        // int colId  = threadIdx.x; // Dense Matrix
        int start_idx = ptrb[rowId];
        int end_idx = ptrb[rowId + 1];


        for(int colId = threadIdx.x; colId < k; colId += blockDim.x){

                float buff = 0.0f;
                for(int i=start_idx; i<end_idx; i++){
                        int offset = indx[i]*k + colId;
                        buff += val[i] * b[offset];
                }
                c[rowId*k + colId] = buff;
        }
}*/
/* v3.0 */
// __global__ void fusedmm_kernel(int m, int n, int k, int nnz,
//                               const int64_t* indx, const int64_t* ptrb, const float* val,
//                               const float* b_transpose, float* c) {

//     int col = blockIdx.x;  // Current column index processed by this block

//     // Loop over the rows of the sparse matrix
//     for (int row = threadIdx.x; row < m; row += blockDim.x) {
//         // Compute the starting and ending indices in the sparse matrix for this row
//         int start = ptrb[row];
//         int end = ptrb[row + 1];

//         float sum = 0.0f;

//         // Loop over the non-zero elements in the row of the sparse matrix
//         for (int i = start; i < end; i++) {
//             int index = indx[i];  // Column index of non-zero element
//             float value = val[i]; // Value of the non-zero element

//             // Perform dot product between the row of the sparse matrix and the column of the transposed dense matrix
//             sum += value * b_transpose[col * n + index];
//         }

//         // Store the computed result in the output matrix
//         c[row * k + col] = sum;
//     }
// }
/* --- v3.0 end--- */

/* v1.1*/
// __global__ void fusedmm_kernel(int m, int n, int k, int nnz, 
//                                const int64_t* indx, const int64_t* ptrb, const float* val, 
//                                const float* b, float* c)
// {
// 	int rowId = blockIdx.x; // Sparse Matrix
// 	// int colId  = threadIdx.x; // Dense Matrix
// 	int start_idx = ptrb[rowId];
// 	int end_idx = ptrb[rowId + 1];
	
// 	__shared__ float sharedVal[256];
// 	__shared__ float sharedCol[256];
	
// 	if(threadIdx.x == 0){
// 		int len = end_idx - start_idx;
//                 int i = 0;
// 		while(i<len){
// 			sharedVal[i] = val[start_idx+i];
// 			sharedCol[i] = indx[start_idx+i];
// 			i++;
// 		}
// 	}
// 	__syncthreads();


// 	for(int colId = threadIdx.x; colId < k; colId += blockDim.x){
		
// 		float buff = 0.0f;
// 		for(int i=start_idx; i<end_idx; i++){
// 			int offset = sharedCol[i-start_idx]*k + colId;
// 			buff += sharedVal[i-start_idx] * b[offset];
// 		}
// 		c[rowId*k + colId] = buff;
// 	}
// }
/* --- v1.1 end ---*/
/*
__global__ void fusedmm_kernel(int m, int n, int k, int nnz,
                              const int64_t* indx, const int64_t* ptrb, const float* val,
                              const float* b_transpose, float* c) {

    extern __shared__ float shared_col[];  // Shared memory for storing the column
    
    int col = blockIdx.x;  // Current column index processed by this block

    // Copy the column from global memory to shared memory
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        shared_col[i] = b_transpose[col * n + i];
    }
    __syncthreads();  // Ensure all threads have finished copying before proceeding

    // Loop over the rows of the sparse matrix
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        // Compute the starting and ending indices in the sparse matrix for this row
        int start = ptrb[row];
        int end = ptrb[row + 1];

        float sum = 0.0f;

        // Loop over the non-zero elements in the row of the sparse matrix
        for (int i = start; i < end; i++) {
            int index = indx[i];  // Column index of non-zero element
            float value = val[i]; // Value of the non-zero element

            // Perform dot product using the column from shared memory
            sum += value * shared_col[index];
        }

        // Store the computed result in the output matrix
        c[row * k + col] = sum;
    }
}
*/


/*v1.1*/
// void fusedmm_spmm_trusted_kernel(int m, int n, int k, int nnz,
//                                const int64_t* indx, const int64_t* ptrb, const float* val,
//                                const float* b, float* c) {

// 	// int num_blocks = k;
//     // int threads_per_block = (m > 1024) ? 1024 : m;


//     // // int nnz_per_block = 128;
//     // // int threads_per_block = 256;
// 	// // int num_blocks = (nnz + nnz_per_block - 1) / nnz_per_block;
//     // fusedmm_kernel<<<num_blocks, threads_per_block>>>(m, n, k, nnz, indx, ptrb, val, b, c);
//     int num_blocks = m;
//   	int threads_per_block = 256;
//   	fusedmm_kernel<<<num_blocks, threads_per_block>>>(m, n, k, nnz, indx, ptrb, val, b, c);
// }
/*<-- v1.1 end -->*/


/*v4*/
// ====
_global_ void fusedmm_kernel(int m, int n, int k, int nnz, 
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
    fusedmm_kernel<<<num_blocks, threads_per_block>>>(m, n, k, nnz, indx, ptrb, val, b, c);
}
/*---v4 end---*/


