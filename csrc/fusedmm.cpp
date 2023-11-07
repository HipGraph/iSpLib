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

#include <limits>
#include <tuple>
#include <stdio.h>
#include <chrono>  
#include <cstdlib>

std::chrono::time_point<std::chrono::system_clock> start;
std::chrono::duration<double> elapsed_seconds;


#include "fusedMM.h"


extern "C" {

void performDummySpMM(int64_t flag);

int fusedMM_csr 
(
   const int32_t imessage,    // message to dictate the operations  
   const INDEXTYPE m,         // number of row of X
   const INDEXTYPE n,         // number of row of Y
   const INDEXTYPE k,         // dimension (col of X or Y)
   const VALUETYPE alpha,     // not used yet
   const INDEXTYPE nnz,       // nonzeros in sparse matrix 
   const INDEXTYPE rows,      // number of rows in sparse matrix
   const INDEXTYPE cols,      // number of columns in sparse matrix 
   const VALUETYPE *val,      // value of non-zeros 
   const INDEXTYPE *indx,     // colids -> column indices 
   const INDEXTYPE *pntrb,    // starting of rowptr for each row
   const INDEXTYPE *pntre,    // ending of rowptr for each row
   const VALUETYPE *x,        // Dense X matrix
   const INDEXTYPE ldx,       // 1eading dimension of X   
   const VALUETYPE *y,        // Dense Y matrix
   const INDEXTYPE ldy,       // leading dimension of Y   
   const VALUETYPE beta,      // beta value 
   VALUETYPE *z,              // Dense matrix Z
   const INDEXTYPE ldz,        // leading dimension size of z 
   INDEXTYPE *z_arg
);

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

}


std::tuple<torch::Tensor, torch::optional<torch::Tensor>> fusedmm_spmm_fw(torch::Tensor rowptr, torch::Tensor col, torch::optional<torch::Tensor> value, torch::Tensor mat, int reduction = 0)
{
    // std::cout << "HEllo!" <<std::endl;
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
    // torch::Tensor out = torch::empty(sizes, mat.options()); -> cannot use empty, Fusedmm expected 0 value. C = alpha.AB + beta.C
    torch::Tensor out;

    if (reduction == 1)       //max
      out = torch::full(sizes, std::numeric_limits<VALUETYPE>::lowest(), mat.options());
    else if (reduction == 2)  //min
      out = torch::full(sizes, std::numeric_limits<VALUETYPE>::max(), mat.options());
    else
      out = torch::zeros(sizes, mat.options());
      
    // torch::Tensor out_arg; // = torch::zeros(sizes, mat.options());
    torch::optional<torch::Tensor> out_arg = torch::nullopt;
    auto mat_a = torch::empty({1});
    VALUETYPE * a = mat_a.data_ptr<VALUETYPE>();
    VALUETYPE * b = mat.data_ptr<VALUETYPE>();
    VALUETYPE * c = out.data_ptr<VALUETYPE>();
    INDEXTYPE * c_idx = 0;

    // mytest_csr(tkern, M, N, K, alpha, S_nnz, S_rows, S_cols, S_values, S_colids, S_rowptr, S_rowptr + 1, a, lda, b, ldb, beta, c, ldc);
    int32_t imsg;
    // if (reduction == 0) //sum
    //  imsg = VOP_COPY_RHS | ROP_NOOP | SOP_COPY | VSC_MUL | AOP_ADD;
    // else if (reduction == 1) //max
    //  imsg = VOP_COPY_RHS | ROP_NOOP | SOP_COPY | VSC_MUL | AOP_MAX;
    switch (reduction){
      case 1: //max
        imsg = VOP_COPY_RHS | ROP_NOOP | SOP_COPY | VSC_MUL | AOP_MAX;
        out_arg = torch::full_like(out, col.numel(), rowptr.options());
        c_idx = out_arg.value().data_ptr<INDEXTYPE>();
        break;
      case 2: //min
        imsg = VOP_COPY_RHS | ROP_NOOP | SOP_COPY | VSC_MUL | AOP_MIN;
        // out = torch::full_like(out, mat.options());
        out_arg = torch::full_like(out, col.numel(), rowptr.options());
        c_idx = out_arg.value().data_ptr<INDEXTYPE>();
        break;
      case 3: //mean
        imsg = VOP_COPY_RHS | ROP_NOOP | SOP_COPY | VSC_MEAN | AOP_ADD;
        break;
      default:  //sum
        imsg = VOP_COPY_RHS | ROP_NOOP | SOP_COPY | VSC_MUL | AOP_ADD;
        break;
    }
    // printf("Hello");
    // std::cout << "imsg: " << imsg << std::endl;
    
	  start = std::chrono::system_clock::now();
    if (rowptr.device().is_cuda())
    {
      std::cout << "Invoking FusedMM CUDA\n" ;
      const char tkern = 'm';
      fusedmm_cuda(tkern, M, N, K, alpha, S_nnz, S_rows, S_cols, S_values, S_colids, S_rowptr, S_rowptr + 1, a, lda, b, ldb, beta, c, ldc);
    }
    else{
      fusedMM_csr(imsg, M, N, K, alpha, S_nnz, S_rows, S_cols, S_values, S_colids, S_rowptr, S_rowptr + 1, a, lda, b, ldb, beta, c, ldc, c_idx);
    }

    if (std::getenv("FUSEDMM_DEBUG_ALL"))
      printf("\nFUSEDMM_ONLY: %.8lf\n", elapsed_seconds.count());
    return std::make_tuple(out, out_arg);
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
                               Variable mat, bool has_value,
                               torch::optional<Variable> value_index_select,
                               torch::optional<Variable> row_index_select) {

    start = std::chrono::system_clock::now();

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
    auto out = std::get<0>(fusedmm_spmm_fw(rowptr, col, opt_value, mat));

    auto value_index_select_ = value_index_select.value();
    auto row_index_select_ = row_index_select.value();

    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward({row, rowptr, col, value, colptr, csr2csc, mat, value_index_select_, row_index_select_});
    
    elapsed_seconds = std::chrono::system_clock::now() - start;
    if (std::getenv("FUSEDMM_DEBUG"))
      printf("\nFUSEDMM_SPMM_SUM_FW: %.8lf\n", elapsed_seconds.count());
    
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    
    start = std::chrono::system_clock::now();

    auto has_value = ctx->saved_data["has_value"].toBool();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto row = saved[0], rowptr = saved[1], col = saved[2], value = saved[3],
         colptr = saved[4], csr2csc = saved[5], mat = saved[6], value_index_select = saved[7], row_index_select = saved[8];

    auto grad_value = Variable();
    if (has_value > 0 && torch::autograd::any_variable_requires_grad({value})) {
      // grad_value = spmm_value_bw_cpu(row, rowptr, col, mat, grad_out, "sum");
      grad_value = Variable();
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      torch::optional<torch::Tensor> opt_value = torch::nullopt;
      if (has_value)
        // opt_value = value.view({-1, 1}).index_select(0, csr2csc).view(-1);
        opt_value = value_index_select;
      else
        opt_value = torch::ones_like(col);

    //   grad_mat = std::get<0>(spmm_fw(colptr, row.index_select(0, csr2csc), opt_value, grad_out, "sum"));
      // v2 grad_mat = std::get<0>(fusedmm_spmm_fw(colptr, row.index_select(0, csr2csc), opt_value, grad_out));
      grad_mat = std::get<0>(fusedmm_spmm_fw(colptr,row_index_select , opt_value, grad_out));
    }

    elapsed_seconds = std::chrono::system_clock::now() - start;
    if (std::getenv("FUSEDMM_DEBUG"))
      printf("\nFUSEDMM_SPMM_SUM_BW: %.8lf\n", elapsed_seconds.count());

    return {Variable(), Variable(), Variable(), grad_value,
            Variable(), Variable(), grad_mat, Variable(), Variable(), Variable()};
  }
};

class FusedMM_SPMMMean : public torch::autograd::Function<FusedMM_SPMMMean> {
public:
  static variable_list forward(AutogradContext *ctx,
                               torch::optional<Variable> opt_row,
                               Variable rowptr, Variable col, Variable value,
                               torch::optional<Variable> opt_rowcount,
                               torch::optional<Variable> opt_colptr,
                               torch::optional<Variable> opt_csr2csc,
                               Variable mat, bool has_value,
                               torch::optional<Variable> new_row,
                               torch::optional<Variable> new_rowcount) {
                              
    start = std::chrono::system_clock::now();
    if (has_value && torch::autograd::any_variable_requires_grad({value})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
    }

    if (torch::autograd::any_variable_requires_grad({mat})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
      AT_ASSERTM(opt_rowcount.has_value(), "Argument `rowcount` is missing");
      AT_ASSERTM(opt_colptr.has_value(), "Argument `colptr` is missing");
      AT_ASSERTM(opt_csr2csc.has_value(), "Argument `csr2csc` is missing");
    }

    auto row = opt_row.has_value() ? opt_row.value() : col;
    auto rowcount = opt_rowcount.has_value() ? opt_rowcount.value() : col;
    auto colptr = opt_colptr.has_value() ? opt_colptr.value() : col;
    auto csr2csc = opt_csr2csc.has_value() ? opt_csr2csc.value() : col;

    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (has_value)
      opt_value = value;
    else
      opt_value = torch::ones_like(col);  

    auto out = std::get<0>(fusedmm_spmm_fw(rowptr, col, opt_value, mat, 3));
    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward({row, rowptr, col, value, rowcount, colptr, csr2csc, mat, new_row.value(), new_rowcount.value()});
    
    elapsed_seconds = std::chrono::system_clock::now() - start;
    if (std::getenv("FUSEDMM_DEBUG"))
      printf("\nFUSEDMM_SPMM_MEAN_FW: %.8lf\n", elapsed_seconds.count());
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    start = std::chrono::system_clock::now();
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto row = saved[0], rowptr = saved[1], col = saved[2], value = saved[3],
         rowcount = saved[4], colptr = saved[5], csr2csc = saved[6],
         mat = saved[7], new_row = saved[8], new_rowcount = saved[9];

    auto grad_value = Variable();
    if (has_value > 0 && torch::autograd::any_variable_requires_grad({value})) {
      // grad_value = spmm_value_bw(row, rowptr, col, mat, grad_out, "mean");
      grad_value = Variable();
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      // row = row.index_select(0, csr2csc);
      // rowcount = rowcount.index_select(0, row).toType(mat.scalar_type());
      // rowcount.masked_fill_(rowcount < 1, 1);

      // if (has_value > 0)
      //   rowcount = value.view({-1, 1}).index_select(0, csr2csc).view(-1).div(rowcount);
      // else
      //   rowcount.pow_(-1);

      // grad_mat = std::get<0>(fusedmm_spmm_fw(colptr, row, rowcount, grad_out)); //sum
      // std::cout << "Calculated:" << std::endl;
      // std::cout << "ROW" << row << std::endl;
      // std::cout << "ROWCOUNT" << rowcount << std::endl;

      // std::cout << "Received:" << std::endl;
      // std::cout << "ROW" << new_row << std::endl;
      // std::cout << "ROWCOUNT" << new_rowcount << std::endl;

      grad_mat = std::get<0>(fusedmm_spmm_fw(colptr, new_row, new_rowcount, grad_out)); //sum
    }
    
    elapsed_seconds = std::chrono::system_clock::now() - start;
    if (std::getenv("FUSEDMM_DEBUG"))
      printf("\nFUSEDMM_SPMM_MEAN_BW: %.8lf\n", elapsed_seconds.count());

    return {Variable(), Variable(), Variable(), grad_value, Variable(),
            Variable(), Variable(), grad_mat,   Variable(), Variable(), Variable()};
  }
};

class FusedMM_SPMMMax : public torch::autograd::Function<FusedMM_SPMMMax> {
public:
  static variable_list forward(AutogradContext *ctx, Variable rowptr,
                               Variable col, Variable value, Variable mat,
                               bool has_value) {

    start = std::chrono::system_clock::now();
    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (has_value)
      opt_value = value;

    auto result = fusedmm_spmm_fw(rowptr, col, opt_value, mat, 1);

    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result).value();
    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward({col, value, mat, arg_out});
    ctx->mark_non_differentiable({arg_out});

    elapsed_seconds = std::chrono::system_clock::now() - start;
    if (std::getenv("FUSEDMM_DEBUG"))
      printf("\nFUSEDMM_SPMM_MAX_FW: %.8lf\n", elapsed_seconds.count());
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    start = std::chrono::system_clock::now();
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto col = saved[0], value = saved[1], mat = saved[2], arg_out = saved[3];

    auto invalid_arg_mask = arg_out == col.size(0);
    arg_out = arg_out.masked_fill(invalid_arg_mask, 0);

    auto grad_value = Variable();
    if (has_value > 0 && torch::autograd::any_variable_requires_grad({value})) {
      auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);
      auto out = mat.gather(-2, ind);
      out.mul_(grad_out);
      out.masked_fill_(invalid_arg_mask, 0);

      grad_value = torch::zeros_like(value);
      grad_value.scatter_add_(0, arg_out.flatten(), out.flatten());
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      if (has_value > 0) {
        value = value.view({-1, 1})
                    .index_select(0, arg_out.flatten())
                    .view_as(arg_out)
                    .mul_(grad_out);
      } else
        value = grad_out;

      value.masked_fill_(invalid_arg_mask, 0);
      auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);

      grad_mat = torch::zeros_like(mat);
      grad_mat.scatter_add_(-2, ind, value);
    }
    
    elapsed_seconds = std::chrono::system_clock::now() - start;
    if (std::getenv("FUSEDMM_DEBUG"))
      printf("\nFUSEDMM_SPMM_MAX_BW: %.8lf\n", elapsed_seconds.count());
    return {Variable(), Variable(), grad_value, grad_mat, Variable()};
  }
};

class FusedMM_SPMMMin : public torch::autograd::Function<FusedMM_SPMMMin> {
public:
  static variable_list forward(AutogradContext *ctx, Variable rowptr,
                               Variable col, Variable value, Variable mat,
                               bool has_value) {

    start = std::chrono::system_clock::now();
    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (has_value)
      opt_value = value;

    auto result = fusedmm_spmm_fw(rowptr, col, opt_value, mat, 2);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result).value();
    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward({col, value, mat, arg_out});
    ctx->mark_non_differentiable({arg_out});
    
    elapsed_seconds = std::chrono::system_clock::now() - start;
    if (std::getenv("FUSEDMM_DEBUG"))
      printf("\nFUSEDMM_SPMM_MIN_FW: %.8lf\n", elapsed_seconds.count());
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    start = std::chrono::system_clock::now();
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto col = saved[0], value = saved[1], mat = saved[2], arg_out = saved[3];

    auto invalid_arg_mask = arg_out == col.size(0);
    arg_out = arg_out.masked_fill(invalid_arg_mask, 0);

    auto grad_value = Variable();
    if (has_value > 0 && torch::autograd::any_variable_requires_grad({value})) {
      auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);
      auto out = mat.gather(-2, ind);
      out.mul_(grad_out);
      out.masked_fill_(invalid_arg_mask, 0);

      grad_value = torch::zeros_like(value);
      grad_value.scatter_add_(0, arg_out.flatten(), out.flatten());
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      if (has_value > 0) {
        value = value.view({-1, 1})
                    .index_select(0, arg_out.flatten())
                    .view_as(arg_out)
                    .mul_(grad_out);
      } else
        value = grad_out;

      value.masked_fill_(invalid_arg_mask, 0);
      auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);

      grad_mat = torch::zeros_like(mat);
      grad_mat.scatter_add_(-2, ind, value);
    }
    elapsed_seconds = std::chrono::system_clock::now() - start;
    if (std::getenv("FUSEDMM_DEBUG"))
      printf("\nFUSEDMM_SPMM_MIN_BW: %.8lf\n", elapsed_seconds.count());
    return {Variable(), Variable(), grad_value, grad_mat, Variable()};
  }
};

SPARSE_API torch::Tensor fusedmm_spmm_add(torch::optional<torch::Tensor> opt_row,
                       torch::Tensor rowptr, torch::Tensor col,
                       torch::optional<torch::Tensor> opt_value,
                       torch::optional<torch::Tensor> opt_colptr,
                       torch::optional<torch::Tensor> opt_csr2csc,
                       torch::Tensor mat, 
                       torch::optional<torch::Tensor> value_index_select,
                       torch::optional<torch::Tensor> row_index_select) {
  auto value = opt_value.has_value() ? opt_value.value() : col;
//   return fusedmm_spmm_fw(rowptr, col, value, mat);
  // std::cout << "HEllo!" <<std::endl;
  // return rowptr;
  return FusedMM_SPMMSum::apply(opt_row, rowptr, col, value, opt_colptr, opt_csr2csc, mat, opt_value.has_value(), value_index_select, row_index_select)[0];
}

SPARSE_API torch::Tensor fusedmm_spmm_mean(torch::optional<torch::Tensor> opt_row,
                       torch::Tensor rowptr, torch::Tensor col,
                       torch::optional<torch::Tensor> opt_value,
                       torch::optional<torch::Tensor> opt_rowcount,
                       torch::optional<torch::Tensor> opt_colptr,
                       torch::optional<torch::Tensor> opt_csr2csc,
                       torch::Tensor mat,
                       torch::optional<torch::Tensor> new_row,
                       torch::optional<torch::Tensor> new_rowcount) {
  auto value = opt_value.has_value() ? opt_value.value() : col;

//   return fusedmm_spmm_fw(rowptr, col, value, mat);
  return FusedMM_SPMMMean::apply(opt_row, rowptr, col, value, opt_rowcount, opt_colptr, opt_csr2csc, mat, opt_value.has_value(), new_row, new_rowcount)[0];
}

SPARSE_API std::tuple<torch::Tensor, torch::Tensor> fusedmm_spmm_max(torch::Tensor rowptr, torch::Tensor col, 
         torch::optional<torch::Tensor> opt_value, torch::Tensor mat) {
  auto value = opt_value.has_value() ? opt_value.value() : col;
  auto result = FusedMM_SPMMMax::apply(rowptr, col, value, mat, opt_value.has_value());
  return std::make_tuple(result[0], result[1]);
}

SPARSE_API std::tuple<torch::Tensor, torch::Tensor>
fusedmm_spmm_min(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> opt_value, torch::Tensor mat) {
  auto value = opt_value.has_value() ? opt_value.value() : col;
  auto result = FusedMM_SPMMMin::apply(rowptr, col, value, mat, opt_value.has_value());
  return std::make_tuple(result[0], result[1]);
}

static auto registry = torch::RegisterOperators()
                           .op("isplib::fusedmm_spmm", &fusedmm_spmm_add)
                           .op("isplib::fusedmm_spmm_mean", &fusedmm_spmm_mean)
                           .op("isplib::fusedmm_spmm_max", &fusedmm_spmm_max)
                           .op("isplib::fusedmm_spmm_min", &fusedmm_spmm_min)
                           .op("isplib::performDummySpMM", &performDummySpMM);
