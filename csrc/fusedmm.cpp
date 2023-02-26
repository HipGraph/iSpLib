#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include <iostream>
// #include "cpu/spmm_cpu.h"


#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__spmm_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__spmm_cpu(void) { return NULL; }
#endif
#endif
#endif


#define INDEXTYPE int64_t
#define VALUETYPE float

extern "C" {
void mytest_csr
(
   const char tkern,       // kernel variations
   const INDEXTYPE m,      // rows of A 
   const INDEXTYPE n,      // rows of B
   const INDEXTYPE k,      // dimension: col of A and B
   const VALUETYPE alpha,  // not used yet  
   const INDEXTYPE nnz,    // nonzeros  
   const INDEXTYPE rows,   // number of rows for sparse matrix 
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
);
void performDummySpMM();
}


torch::Tensor fusedmm_spmm(torch::Tensor rowptr, torch::Tensor col, torch::Tensor value, torch::Tensor mat)
{
    VALUETYPE alpha = 1;
    VALUETYPE beta = 0;
    const char tkern = 'm';

    INDEXTYPE M = rowptr.numel() - 1;
    INDEXTYPE N = mat.size(-2);
    INDEXTYPE K = mat.size(-1);

    INDEXTYPE S_rows = M;
    INDEXTYPE S_cols = N;
    INDEXTYPE S_nnz = value.numel();

    INDEXTYPE * S_rowptr = rowptr.data_ptr<INDEXTYPE>();
    INDEXTYPE * S_colids = col.data_ptr<INDEXTYPE>();
    VALUETYPE * S_values = value.data_ptr<VALUETYPE>();
    
    INDEXTYPE lda = K; INDEXTYPE ldb = K; INDEXTYPE ldc = K;

    auto szA = 1;
    auto szB = N * K;
    auto szC = M * K;

    mat = mat.contiguous();

    auto sizes = mat.sizes().vec();
    sizes[mat.dim() - 2] = rowptr.numel() - 1;
    // torch::Tensor out = torch::empty(sizes, mat.options());
    torch::Tensor out = torch::zeros(sizes, mat.options());

    auto mat_a = torch::empty({1});
    VALUETYPE * a = mat_a.data_ptr<VALUETYPE>();
    VALUETYPE * b = mat.data_ptr<VALUETYPE>();
    VALUETYPE * c = out.data_ptr<VALUETYPE>();

    // printf("Sparse Values:\n");
    // for (int i = 0; i< S_nnz; i++)
    //     std::cout << S_values[i] << ",";
    // printf("\n\n");

    // printf("Dense Values:\n");
    // for (int i = 0; i< szB; i++)
    //     std::cout << b[i] << ",";
    // printf("\n");

    mytest_csr(tkern, M, N, K, alpha, S_nnz, S_rows, S_cols, S_values, S_colids, S_rowptr, S_rowptr + 1, a, lda, b, ldb, beta, c, ldc);
    
    return out;
    // auto col_data = col.data_ptr<int64_t>();

    // auto B = mat.numel() / (N * K);
    /*
    printf("Running fusedmm! Printing rowptr values...\n");
    for (int i = 0; i< M; i++)
        std::cout << rowptr_data[i] << ",";
    printf("\n");
    */
}

static auto registry = torch::RegisterOperators()
                           .op("torch_sparse::fusedmm_spmm", &fusedmm_spmm)
                           .op("torch_sparse::performDummySpMM", &performDummySpMM);
