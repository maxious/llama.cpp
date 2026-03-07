#include "../fattn-fused.hpp"

template void ggml_sycl_flash_attn_ext_fused_case<80, GGML_TYPE_F16, GGML_TYPE_F16>(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
