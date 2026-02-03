// fattn_xmx.hpp - XMX Flash Attention Declarations  
// Reduced from implementation header during file split

#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "fattn_common.hpp"
#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_COOPERATIVE_MATRICES

void ggml_sycl_op_flash_attn_coopmat_padded(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_flash_attn_coopmat_direct(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_flash_attn_coopmat_kvsplit(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // SYCL_EXT_COOPERATIVE_MATRICES

#endif // GGML_SYCL_FATTN_XMX_HPP
