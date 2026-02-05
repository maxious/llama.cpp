//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <sys/time.h>

#include <sycl/sycl.hpp>
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#include <sycl/half_type.hpp>

#include "ggml-sycl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/gemm_tiled.hpp"
#include "ggml-sycl/gemm_f16_f32_tiled.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml.h"
#include "sycl_defs.hpp"

bool g_sycl_loaded = false;
int g_ggml_sycl_debug = 0;
int g_ggml_sycl_disable_optimize = 0;
int g_ggml_sycl_disable_graph = 0;
int g_ggml_sycl_disable_dnn = 0;
int g_ggml_sycl_prioritize_dmmv = 0;
int g_ggml_sycl_use_async_mem_op = 0;

static ggml_sycl_device_info ggml_sycl_init() {
    ggml_sycl_device_info info = {};

    info.device_count = dpct::dev_mgr::instance().device_count();
    if (info.device_count == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    for (int i = 0; i < info.device_count; ++i) {
        info.devices[i].vmm = 0;
        dpct::device_info prop;
        sycl::device device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm = prop.get_max_compute_units();
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
        info.devices[i].smpbo = prop.get_local_mem_size();

        // Determine MMQ tile configuration based on device architecture
        // Using version numbers and vendor ID for maximum compatibility
        mmq_config cfg = {64, 128, 4};
        sycl_arch_type arch_type = SYCL_ARCH_UNKNOWN;

        const int major = prop.get_major_version();
        const int minor = prop.get_minor_version();
        const uint32_t vendor_id = device.get_info<sycl::info::device::vendor_id>();

        if (vendor_id == 0x8086) {
            if (major >= 13) {
                arch_type = SYCL_ARCH_INTEL_XE2;
                cfg = {64, 128, 8};
            } else if (major == 12) {
                arch_type = SYCL_ARCH_INTEL_XE;
                cfg = {64, 128, 4};
            } else if (major >= 9) {
                arch_type = SYCL_ARCH_INTEL_GEN9;
            }
         } else if (vendor_id == 0x1002) {
             if (major >= 11) {
                 arch_type = SYCL_ARCH_AMD_RDNA3;
                 cfg = {64, 128, 8};
            } else if (major == 10) {
                if (minor >= 30) {
                    arch_type = SYCL_ARCH_AMD_RDNA2;
                    cfg = {64, 128, 8};
                } else {
                    arch_type = SYCL_ARCH_AMD_RDNA1;
                    cfg = {64, 64, 8};
                }
            } else {
                arch_type = SYCL_ARCH_UNKNOWN;
                cfg = {64, 128, 4};
            }
          } else if (vendor_id == 0x10de) {
              if (major == 7 && minor >= 5) {
                  arch_type = SYCL_ARCH_NVIDIA_TURING;
                  cfg = {64, 128, 4};  // Same tile config as Ampere for now
              } else if (major >= 8) {
                  arch_type = SYCL_ARCH_NVIDIA_AMPERE;
                  cfg = {64, 128, 4};
              } else {
                  arch_type = SYCL_ARCH_NVIDIA_AMPERE;  // fallback
                  cfg = {64, 128, 4};
              }
           }

        const char* force_arch = std::getenv("GGML_SYCL_FORCE_ARCH");
        if (force_arch && std::strlen(force_arch) > 0) {
            if (std::strcmp(force_arch, "INTEL_XE2") == 0) {
                arch_type = SYCL_ARCH_INTEL_XE2;
                cfg = {64, 128, 8};
            } else if (std::strcmp(force_arch, "INTEL_XE") == 0) {
                arch_type = SYCL_ARCH_INTEL_XE;
                cfg = {64, 128, 4};
            } else if (std::strcmp(force_arch, "INTEL_GEN9") == 0) {
                arch_type = SYCL_ARCH_INTEL_GEN9;
                cfg = {64, 128, 4};
            } else if (std::strcmp(force_arch, "AMD_RDNA3") == 0) {
                arch_type = SYCL_ARCH_AMD_RDNA3;
                cfg = {64, 128, 8};
            } else if (std::strcmp(force_arch, "AMD_RDNA2") == 0) {
                arch_type = SYCL_ARCH_AMD_RDNA2;
                cfg = {64, 128, 8};
            } else if (std::strcmp(force_arch, "AMD_RDNA1") == 0) {
                arch_type = SYCL_ARCH_AMD_RDNA1;
                cfg = {64, 64, 8};
            } else if (std::strcmp(force_arch, "NVIDIA_AMPERE") == 0) {
                arch_type = SYCL_ARCH_NVIDIA_AMPERE;
                cfg = {64, 128, 4};
            } else if (std::strcmp(force_arch, "NVIDIA_TURING") == 0) {
                arch_type = SYCL_ARCH_NVIDIA_TURING;
                cfg = {64, 128, 4};
            } else {
                std::cerr << "Warning: Unknown GGML_SYCL_FORCE_ARCH value: " << force_arch << ". Using detected configuration.\n";
            }
        }

        info.devices[i].arch = arch_type;
        info.devices[i].mmq = cfg;

        GGML_LOG_INFO("Device %d: vendor=0x%04x, arch=%d, mmq={%d,%d,%d}\n",
                      i, vendor_id, arch_type, cfg.mmq_x, cfg.mmq_y, cfg.nwarps);

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }
    return info;
}

const ggml_sycl_device_info & ggml_sycl_info() {
    static ggml_sycl_device_info info = ggml_sycl_init();
    return info;
}

static void print_device_detail(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

static void print_device_opt_feature(int device_count) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_count);

    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |Max    |        |Max  |Global |                     |\n");
    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |compute|Max work|sub  |mem    |                     |\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|                                   "
        "Name|Version|units  |group   |group|size   |       Driver version|\n");
    GGML_LOG_INFO(
        "|--|-------------------|---------------------------------------|------"
        "-|-------|--------|-----|-------|---------------------|\n");

    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_count);
}


void ggml_check_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        g_ggml_sycl_debug = get_sycl_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_disable_optimize = get_sycl_env("GGML_SYCL_DISABLE_OPT", 0);
        g_ggml_sycl_disable_graph = get_sycl_env("GGML_SYCL_DISABLE_GRAPH", 1);
        g_ggml_sycl_disable_dnn = get_sycl_env("GGML_SYCL_DISABLE_DNN", 0);
        g_ggml_sycl_prioritize_dmmv = get_sycl_env("GGML_SYCL_PRIORITIZE_DMMV", 0);
        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_OPT: %d\n", g_ggml_sycl_disable_optimize);
#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_GRAPH: %d\n", g_ggml_sycl_disable_graph);
#else
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_GRAPH: graph disabled by compile flag\n");
#endif
#if GGML_SYCL_DNNL
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_DNN: %d\n", g_ggml_sycl_disable_dnn);
#else
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_DNN: DNN disabled by compile flag\n");
#endif
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);
        GGML_LOG_INFO("Build with Macros:\n");
#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif
#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

/* NOT REMOVE, keep it for next optimize for XMX.
#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
*/
        // Currently, we only use async malloc / free when graphs are enabled as it is required for the calls to be
        // properly recorded. As this SYCL extension matures it may be beneficial to enable as the default path and in
        // other places.
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = !g_ggml_sycl_disable_graph;
        if (g_ggml_sycl_use_async_mem_op) {
            for (unsigned int i = 0; i < dpct::dev_mgr::instance().device_count(); ++i) {
                if (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    break;
                }
            }
        }
#endif
        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                            dpct::dev_mgr::instance().device_count()) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        initialized = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/

GGML_API void ggml_backend_sycl_get_gpu_list(int *id_list, int max_len) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    for(int i=0;i<max_len;i++) id_list[i] = -1;

    for (int i=0;i< ggml_sycl_info().device_count;i++){
        if (i>=max_len) break;
        id_list[i] = i;
    }
    return;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

