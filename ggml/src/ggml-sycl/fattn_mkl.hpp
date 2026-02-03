// fattn_mkl.hpp - oneMKL Flash Attention Declarations
// Reduced from implementation header during file split

#ifdef GGML_SYCL_USE_INTEL_ONEMKL

#include "fattn_common.hpp"

template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_USE_INTEL_ONEMKL
