// fattn_mkl.hpp - oneMKL Flash Attention Declarations
// Reduced from implementation header during file split

#include "fattn_common.hpp"

template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
