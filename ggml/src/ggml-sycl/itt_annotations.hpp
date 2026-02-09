//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_ITT_ANNOTATIONS_HPP
#define GGML_SYCL_ITT_ANNOTATIONS_HPP

// ============================================================================
// SYCL ITT (Instrumentation and Tracing Technology) Annotations
// ============================================================================
//
// This header provides ITT annotation macros for profiling SYCL backend ops.
// ITT annotations allow VTune and other Intel tools to display semantic
// regions in the timeline, making it easier to understand which kernel
// implementations are being used.
//
// Enable ITT annotations by setting the environment variable:
//   export INTEL_ENABLE_OFFLOAD_ANNOTATIONS=1
//
// Or by building with:
//   -DGGML_SYCL_ITT_ANNOTATIONS=1
//
// The DPC++ compiler automatically instruments device code via the
// SPIRITTAnnotations LLVM pass when the above env var is set.
//
// For host-side annotations, we use the ittnotify library directly.
//
// ============================================================================

#include <cstdlib>
#include <cstring>

// Check if ITT annotations are enabled
#if defined(GGML_SYCL_ITT_ANNOTATIONS)
#    define GGML_SYCL_ITT_ENABLED 1
#elif defined(INTEL_ENABLE_OFFLOAD_ANNOTATIONS)
#    define GGML_SYCL_ITT_ENABLED 1
#else
// Check environment variable at compile time (for header-only check)
// Runtime check is also performed in the implementation
#    define GGML_SYCL_ITT_ENABLED 0
#endif

// ============================================================================
// Host-side ITT annotations using ittnotify
// ============================================================================

#if GGML_SYCL_ITT_ENABLED
// Include the ITT notify header if available
#    if __has_include(<ittnotify.h>)
#        include <ittnotify.h>
#        define GGML_SYCL_ITT_HAS_HEADER 1
#    elif __has_include(<intel/ittnotify.h>)
#        include <intel/ittnotify.h>
#        define GGML_SYCL_ITT_HAS_HEADER 1
#    else
#        define GGML_SYCL_ITT_HAS_HEADER 0
#        warning "ITT annotations enabled but ittnotify.h not found. Install Intel VTune or oneAPI for full support."
#    endif
#endif

namespace ggml_sycl_itt {

// ============================================================================
// ITT Domain and Task Management
// ============================================================================

#if GGML_SYCL_ITT_ENABLED && GGML_SYCL_ITT_HAS_HEADER

class ITTDomain {
  public:
    explicit ITTDomain(const char * name) : domain(__itt_domain_create(name)) {}

    __itt_domain * get() const { return domain; }

  private:
    __itt_domain * domain;
};

// Global domain for SYCL backend operations
inline __itt_domain * get_sycl_domain() {
    static ITTDomain sycl_domain("ggml_sycl");
    return sycl_domain.get();
}

// String handle cache for common task names
inline __itt_string_handle * get_string_handle(const char * str) {
    static ITTDomain domain("ggml_sycl_strings");
    return __itt_string_handle_create(str);
}

// ============================================================================
// Task Scoping Class
// ============================================================================

class ITTTask {
  public:
    ITTTask(__itt_domain * domain, __itt_string_handle * name) : domain_(domain), active_(true) {
        __itt_task_begin(domain_, __itt_null, __itt_null, name);
    }

    ITTTask(const char * name) : domain_(get_sycl_domain()), active_(true) {
        __itt_task_begin(domain_, __itt_null, __itt_null, get_string_handle(name));
    }

    ~ITTTask() { end(); }

    void end() {
        if (active_) {
            __itt_task_end(domain_);
            active_ = false;
        }
    }

  private:
    __itt_domain * domain_;
    bool           active_;
};

// ============================================================================
// Inline functions for C-style usage
// ============================================================================

inline bool is_enabled() {
    static bool checked = false;
    static bool enabled = false;
    if (!checked) {
        const char * env = std::getenv("INTEL_ENABLE_OFFLOAD_ANNOTATIONS");
        enabled          = (env != nullptr && std::strlen(env) > 0);
        checked          = true;
    }
    return enabled;
}

inline void task_begin(const char * name) {
    if (is_enabled()) {
        __itt_task_begin(get_sycl_domain(), __itt_null, __itt_null, get_string_handle(name));
    }
}

inline void task_end() {
    if (is_enabled()) {
        __itt_task_end(get_sycl_domain());
    }
}

#else   // GGML_SYCL_ITT_ENABLED && GGML_SYCL_ITT_HAS_HEADER

// No-op implementations when ITT is disabled or header not available
class ITTTask {
  public:
    ITTTask(const char *) {}

    ITTTask(void *, void *) {}

    void end() {}
};

inline bool is_enabled() {
    return false;
}

inline void task_begin(const char *) {}

inline void task_end() {}

#endif  // GGML_SYCL_ITT_ENABLED && GGML_SYCL_ITT_HAS_HEADER

}  // namespace ggml_sycl_itt

// ============================================================================
// Convenience Macros
// ============================================================================

#if GGML_SYCL_ITT_ENABLED && GGML_SYCL_ITT_HAS_HEADER

// Scoped task that automatically ends when scope exits
#    define GGML_SYCL_ITT_TASK(name) ggml_sycl_itt::ITTTask _itt_task_##__LINE__(name)

// Begin a task (must pair with GGML_SYCL_ITT_TASK_END)
#    define GGML_SYCL_ITT_TASK_BEGIN(name) ggml_sycl_itt::task_begin(name)

// End a task
#    define GGML_SYCL_ITT_TASK_END() ggml_sycl_itt::task_end()

// Scoped task with custom domain
#    define GGML_SYCL_ITT_TASK_DOMAIN(domain, name) ggml_sycl_itt::ITTTask _itt_task_##__LINE__(domain, name)

#else

// No-op macros when ITT is disabled
#    define GGML_SYCL_ITT_TASK(name)                ((void) 0)
#    define GGML_SYCL_ITT_TASK_BEGIN(name)          ((void) 0)
#    define GGML_SYCL_ITT_TASK_END()                ((void) 0)
#    define GGML_SYCL_ITT_TASK_DOMAIN(domain, name) ((void) 0)

#endif  // GGML_SYCL_ITT_ENABLED && GGML_SYCL_ITT_HAS_HEADER

// ============================================================================
// Flash Attention Specific Macros
// ============================================================================

#define GGML_SYCL_ITT_FATTN_XMX(head_size)   GGML_SYCL_ITT_TASK("fattn:xmx:h" #head_size)
#define GGML_SYCL_ITT_FATTN_FUSED(head_size) GGML_SYCL_ITT_TASK("fattn:fused:h" #head_size)
#define GGML_SYCL_ITT_FATTN_TILED(head_size) GGML_SYCL_ITT_TASK("fattn:tiled:h" #head_size)
#define GGML_SYCL_ITT_FATTN_MKL(head_size)   GGML_SYCL_ITT_TASK("fattn:mkl:h" #head_size)

// Variant with dynamic name (for templates)
#define GGML_SYCL_ITT_FATTN_XMX_DYNAMIC()   GGML_SYCL_ITT_TASK("fattn:xmx")
#define GGML_SYCL_ITT_FATTN_FUSED_DYNAMIC() GGML_SYCL_ITT_TASK("fattn:fused")
#define GGML_SYCL_ITT_FATTN_TILED_DYNAMIC() GGML_SYCL_ITT_TASK("fattn:tiled")
#define GGML_SYCL_ITT_FATTN_MKL_DYNAMIC()   GGML_SYCL_ITT_TASK("fattn:mkl")

// ============================================================================
// Matrix Multiplication Specific Macros
// ============================================================================

#define GGML_SYCL_ITT_MUL_MAT_MMQ(type)   GGML_SYCL_ITT_TASK("mul_mat:mmq:" #type)
#define GGML_SYCL_ITT_MUL_MAT_XMX(type)   GGML_SYCL_ITT_TASK("mul_mat:xmx:" #type)
#define GGML_SYCL_ITT_MUL_MAT_MKL(type)   GGML_SYCL_ITT_TASK("mul_mat:mkl:" #type)
#define GGML_SYCL_ITT_MUL_MAT_TILED(type) GGML_SYCL_ITT_TASK("mul_mat:tiled:" #type)

// MUL_MAT_ID (MoE) variants
#define GGML_SYCL_ITT_MUL_MAT_ID_MMQ(type)   GGML_SYCL_ITT_TASK("mul_mat_id:mmq:" #type)
#define GGML_SYCL_ITT_MUL_MAT_ID_TILED(type) GGML_SYCL_ITT_TASK("mul_mat_id:tiled:" #type)

// ============================================================================
// General Operation Macros
// ============================================================================

#define GGML_SYCL_ITT_OP(name)       GGML_SYCL_ITT_TASK("op:" #name)
#define GGML_SYCL_ITT_OP_BEGIN(name) GGML_SYCL_ITT_TASK_BEGIN("op:" #name)
#define GGML_SYCL_ITT_OP_END()       GGML_SYCL_ITT_TASK_END()

// Architecture-specific markers
#define GGML_SYCL_ITT_ARCH_DETECTED(arch) GGML_SYCL_ITT_TASK("arch:" #arch)

#endif  // GGML_SYCL_ITT_ANNOTATIONS_HPP
