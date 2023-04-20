#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include <iostream>
#include "cpu/spmm_cpu.cpp"

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

void cuda_spmm_test();
void fusedmm_cuda
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

torch::Tensor fusedmm_spmm_fw(torch::Tensor rowptr, torch::Tensor col, torch::optional<torch::Tensor> value, torch::Tensor mat)
{
    VALUETYPE alpha = 1;
    VALUETYPE beta = 0;
    const char tkern = 'm';

    INDEXTYPE M = rowptr.numel() - 1;
    INDEXTYPE N = mat.size(-2);
    INDEXTYPE K = mat.size(-1);

    INDEXTYPE S_rows = M;
    INDEXTYPE S_cols = N;
    INDEXTYPE S_nnz = value.value().numel();

    INDEXTYPE * S_rowptr = rowptr.data_ptr<INDEXTYPE>();
    INDEXTYPE * S_colids = col.data_ptr<INDEXTYPE>();
    // VALUETYPE * S_values = value.data_ptr<VALUETYPE>();
    VALUETYPE * S_values = value.value().data_ptr<VALUETYPE>();

    
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

    if (rowptr.device().is_cuda())
    {
      std::cout << "Invoking FusedMM CUDA\n" ;
      fusedmm_cuda(tkern, M, N, K, alpha, S_nnz, S_rows, S_cols, S_values, S_colids, S_rowptr, S_rowptr + 1, a, lda, b, ldb, beta, c, ldc);
    }
    else{
      mytest_csr(tkern, M, N, K, alpha, S_nnz, S_rows, S_cols, S_values, S_colids, S_rowptr, S_rowptr + 1, a, lda, b, ldb, beta, c, ldc);
    }
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


using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class FusedMM_SPMMSum : public torch::autograd::Function<FusedMM_SPMMSum> {
public:
  static variable_list forward(AutogradContext *ctx,
                               torch::optional<Variable> opt_row,
                               Variable rowptr, Variable col, Variable value,
                               torch::optional<Variable> opt_colptr,
                               torch::optional<Variable> opt_csr2csc,
                               Variable mat, bool has_value) {

    if (has_value && torch::autograd::any_variable_requires_grad({value})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
    }

    if (torch::autograd::any_variable_requires_grad({mat})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
      AT_ASSERTM(opt_colptr.has_value(), "Argument `colptr` is missing");
      AT_ASSERTM(opt_csr2csc.has_value(), "Argument `csr2csc` is missing");
    }

    auto row = opt_row.has_value() ? opt_row.value() : col;
    auto colptr = opt_colptr.has_value() ? opt_colptr.value() : col;
    auto csr2csc = opt_csr2csc.has_value() ? opt_csr2csc.value() : col;

    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (has_value)
      opt_value = value;
    else
      opt_value = torch::ones_like(col);  

    // auto out = std::get<0>(spmm_fw(rowptr, col, opt_value, mat, "sum"));
    auto out = fusedmm_spmm_fw(rowptr, col, opt_value, mat);

    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward({row, rowptr, col, value, colptr, csr2csc, mat});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto row = saved[0], rowptr = saved[1], col = saved[2], value = saved[3],
         colptr = saved[4], csr2csc = saved[5], mat = saved[6];

    auto grad_value = Variable();
    if (has_value > 0 && torch::autograd::any_variable_requires_grad({value})) {
      // grad_value = spmm_value_bw_cpu(row, rowptr, col, mat, grad_out, "sum");
      grad_value = Variable();

    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      torch::optional<torch::Tensor> opt_value = torch::nullopt;
      if (has_value)
        opt_value = value.view({-1, 1}).index_select(0, csr2csc).view(-1);
      else
        opt_value = torch::ones_like(col);

    //   grad_mat = std::get<0>(spmm_fw(colptr, row.index_select(0, csr2csc), opt_value, grad_out, "sum"));
      grad_mat = fusedmm_spmm_fw(colptr, row.index_select(0, csr2csc), opt_value, grad_out);

    }

    return {Variable(), Variable(), Variable(), grad_value,
            Variable(), Variable(), grad_mat,   Variable()};
  }
};


SPARSE_API torch::Tensor fusedmm_spmm(torch::optional<torch::Tensor> opt_row,
                       torch::Tensor rowptr, torch::Tensor col,
                       torch::optional<torch::Tensor> opt_value,
                       torch::optional<torch::Tensor> opt_colptr,
                       torch::optional<torch::Tensor> opt_csr2csc,
                       torch::Tensor mat) {
  auto value = opt_value.has_value() ? opt_value.value() : col;
//   return fusedmm_spmm_fw(rowptr, col, value, mat);
  return FusedMM_SPMMSum::apply(opt_row, rowptr, col, value, opt_colptr, opt_csr2csc,
                        mat, opt_value.has_value())[0];
}

void test_cuda()
{
  cuda_spmm_test();
}

static auto registry = torch::RegisterOperators()
                           .op("isplib::fusedmm_spmm", &fusedmm_spmm)
                           .op("isplib::test_cuda", &test_cuda)
                           .op("isplib::performDummySpMM", &performDummySpMM);
