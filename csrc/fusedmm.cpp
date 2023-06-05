#pragma once

#ifdef _WIN32
#if defined(torchsparse_EXPORTS)
#define SPARSE_API __declspec(dllexport)
#else
#define SPARSE_API __declspec(dllimport)
#endif
#else
#define SPARSE_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define SPARSE_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define SPARSE_INLINE_VARIABLE __declspec(selectany)
#else
#define SPARSE_INLINE_VARIABLE __attribute__((weak))
#endif
#endif

#ifdef WITH_PYTHON
#include <Python.h>
#endif

#include <ATen/Parallel.h>
#include <torch/script.h>
#include <iostream>
// #include "cpu/spmm_cpu.cpp"

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

// torch::Tensor spmm_value_bw_cpu(torch::Tensor row, torch::Tensor rowptr,
//                                 torch::Tensor col, torch::Tensor mat,
//                                 torch::Tensor grad, std::string reduce) {
//   CHECK_CPU(row);
//   CHECK_CPU(rowptr);
//   CHECK_CPU(col);
//   CHECK_CPU(mat);
//   CHECK_CPU(grad);

//   mat = mat.contiguous();
//   grad = grad.contiguous();

//   auto M = grad.size(-2);
//   auto N = mat.size(-2);
//   auto E = row.numel();
//   auto K = mat.size(-1);
//   auto B = mat.numel() / (N * K);

//   auto out = torch::zeros({row.numel()}, grad.options());

//   auto row_data = row.data_ptr<int64_t>();
//   auto rowptr_data = rowptr.data_ptr<int64_t>();
//   auto col_data = col.data_ptr<int64_t>();
//   AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "spmm_value_bw_cpu", [&] {
//     auto mat_data = mat.data_ptr<scalar_t>();
//     auto grad_data = grad.data_ptr<scalar_t>();
//     auto out_data = out.data_ptr<scalar_t>();

//     scalar_t val;
//     int64_t row, col;
//     AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
//       for (int b = 0; b < B; b++) {
//         for (int e = 0; e < E; e++) {
//           row = row_data[e], col = col_data[e], val = (scalar_t)0;
//           for (int k = 0; k < K; k++) {
//             val += mat_data[b * N * K + col * K + k] *
//                    grad_data[b * M * K + row * K + k];
//           }
//           if (REDUCE == MEAN) {
//             int row_start = rowptr_data[row], row_end = rowptr_data[row + 1];
//             val /= (scalar_t)std::max(row_end - row_start, 1);
//           }
//           out_data[e] += val;
//         }
//       }
//     });
//   });

//   return out;
// }


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

static auto registry = torch::RegisterOperators()
                           .op("isplib::fusedmm_spmm", &fusedmm_spmm)
                           .op("isplib::performDummySpMM", &performDummySpMM);
