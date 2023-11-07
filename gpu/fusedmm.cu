// #include "kernel.cuh"
#define INDEXTYPE int64_t
#define VALUETYPE float

#include "kernels/spmm.cuh"

#include <iostream>
#include <cuda_runtime.h>

#include <chrono>
using namespace std::chrono;
// typedef std::ratio<1l, 1000000000000l> pico;
// typedef duration<long long, pico> picoseconds;

/*void fusedmm_cuda(int m, int n, int k, int nnz,
                               const int64_t* indx, const int64_t* ptrb, const float* val,
                               const float* b, float* c) */
void fusedmm_cuda
(
   const char tkern,       // kernel variations
   const INDEXTYPE m,      // rows of A 
   const INDEXTYPE n,      // rows of B
   const INDEXTYPE k,      // dimension: col of A and B
   const VALUETYPE alpha,  // not used yet  
   const INDEXTYPE nnz,    // nonzeros  
   const INDEXTYPE rows,   // number of rows for sparse matrix (COO format!)
   const INDEXTYPE cols,   // number of columns for sparse matrix 
   const VALUETYPE *val,   // NNZ value  
   const INDEXTYPE *indx,  // colids -> column indices 
   const INDEXTYPE *pntrb, // starting index for rowptr
   const INDEXTYPE *pntre, // ending index for rowptr
   const VALUETYPE *a,     // Dense A (X) matrix
   const INDEXTYPE lda,    // leading dimension of A (col size since A row-major)  
   const VALUETYPE *b,     // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of B (col size since B row-major)  
   const VALUETYPE beta,   // beta value 
   VALUETYPE *c,           // Dense matrix c
   const INDEXTYPE ldc     // leading dimension of c (col size since C row-major) 
)
{

  // Compute the number of threads per block and blocks per grid
  //int row_per_block = 256/k;
  //int n_block = (m+row_per_block-1)/row_per_block;

  // Launch the CUDA kernel
  //fusedmm_spmm_trusted_kernel<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(m, n, k, nnz, indx, pntrb, val, b, c);
  fusedmm_spmm_trusted_kernel(m, n, k, nnz, indx, pntrb, val, b, c);
  // For Cora, best is 102
  // gfusedMM_spmm[102 - 1]('m', m, n, k, 1, nnz, 0, 0,val, indx, pntrb,pntre , 0, 0,  b, 0, 0, c, 0);
}


auto cuda_spmm_test(int kk)
{
    
    // #include "cora.cuh"
    // #include "datasets/photo.cuh"
    
    int64_t *ptrb = new int64_t[64]{
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63};
    
    int64_t *indx = new int64_t[64]{
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63};
    
    float *val = new float[64]{
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    };

    float mat[16][16] = {
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 30.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 40.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 50.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 60.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 70.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 80.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 90.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 110.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 120.0, 10.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 130.0, 10.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 140.0, 10.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 150.0, 10.0},
                        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 160.0}};
    float out[16][16] = {
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
                        };
    int m=16;
    int n=16;
    int k=16;
    int nnz=64;
    int rows=m;
    int cols=n;
    
    
    // Allocate device (GPU) memory
    int64_t *ptrb_device, *indx_device; 
    float *val_device, *mat_device, *out_device;
    cudaMalloc(&ptrb_device, nnz * sizeof(int64_t));
    cudaMalloc(&indx_device, nnz * sizeof(int64_t));
    cudaMalloc(&val_device, nnz * sizeof(float));
    cudaMalloc(&mat_device, n * k * sizeof(float));
    cudaMalloc(&out_device, m * k * sizeof(float)); 
    
    // Copy input data from host to device
    cudaMemcpy(ptrb_device, ptrb, nnz * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(indx_device, indx, nnz * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(val_device, val, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mat_device, mat, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_device, out, m * k * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    // fusedmm_cuda('m', m, n, k, 1, nnz, 0, 0,val_device, indx_device, ptrb_device,ptrb_device+1 , 0, 0,  mat_device, 0, 0, out_device, 0);
    // std::cout << "Invoking generted kernel...\n";
    auto start = high_resolution_clock::now();
    // gfusedMM_spmm[kk - 1]('m', m, n, k, 1, nnz, 0, 0,val_device, indx_device, ptrb_device,ptrb_device+1 , 0, 0,  mat_device, 0, 0, out_device, 0);
    fusedmm_cuda('m', m, n, k, 1, nnz, 0, 0,val_device, indx_device, ptrb_device,ptrb_device+1 , 0, 0,  mat_device, 0, 0, out_device, 0);
    auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<nanoseconds>(stop - start);
    auto duration = duration_cast<microseconds>(stop - start);

    // Copy output data from device to host
    cudaMemcpy(out, out_device, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    
    //Print output
    /*for (int i = 0; i < m; i++) {
	    for(int j=0; j<k; j++){
        	std::cout << out[i][j] << " ";
	    }
	    std::cout << "\n";
    }
    std::cout << std::endl;*/

    // Free memory
    cudaFree(ptrb_device);
    cudaFree(indx_device);
    cudaFree(val_device);
    cudaFree(mat_device);
    cudaFree(out_device);
    delete [] ptrb;
    delete [] indx;
    delete [] val;
    //return 0;
    return duration.count();
}
