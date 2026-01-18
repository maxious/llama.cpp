#ifndef GGML_SYCL_FATTN_COMMON_HPP
#define GGML_SYCL_FATTN_COMMON_HPP

#include <sycl/sycl.hpp>

template<int N>
inline void ggml_sycl_memcpy(float* dst, const float* src) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        dst[i] = src[i];
    }
}

// Vectorized load for memory bandwidth optimization
// Loads 4 floats at once using sycl::vec
inline void ggml_sycl_load_vec4(sycl::float4& dst, const float* src) {
    dst = sycl::float4(src[0], src[1], src[2], src[3]);
}

inline void ggml_sycl_store_vec4(float* dst, const sycl::float4& src) {
    dst[0] = src.x();
    dst[1] = src.y();
    dst[2] = src.z();
    dst[3] = src.w();
}

// Vectorized load for half precision (FP16/BF16) - 4 elements = 8 bytes
inline void ggml_sycl_load_vec4_half(sycl::float4& dst, const sycl::half* src) {
    sycl::half h0 = src[0], h1 = src[1], h2 = src[2], h3 = src[3];
    dst = sycl::float4(static_cast<float>(h0), static_cast<float>(h1),
                       static_cast<float>(h2), static_cast<float>(h3));
}

#endif // GGML_SYCL_FATTN_COMMON_HPP
