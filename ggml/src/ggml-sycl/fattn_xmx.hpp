// fattn_xmx.hpp - XMX Flash Attention Declarations  
// Reduced from implementation header during file split

#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "fattn_common.hpp"
#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_ONEAPI_MATRIX

template<int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM>
void ggml_sycl_op_flash_attn_coopmat_padded(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

template<int64_t DQK, int64_t DV, int BLOCK_M = 32, int BLOCK_N = 32>
void ggml_sycl_op_flash_attn_coopmat_direct(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

template<int64_t HEAD_DIM, int64_t V_HEAD_DIM, int TM, int TN, int TK, int BLOCK_M, int BLOCK_N, fattn_input_type InputType, bool V_FROM_K>
void ggml_sycl_op_flash_attn_coopmat_kvsplit(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // SYCL_EXT_ONEAPI_MATRIX

#endif // GGML_SYCL_FATTN_XMX_HPP
