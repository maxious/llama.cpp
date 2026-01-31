#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

set(TARGET llama-ls-sycl-device)
add_executable(${TARGET} ls-sycl-device.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-sycl-fattn-proto)
add_executable(${TARGET} fattn-proto.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common ggml llama ${CMAKE_THREAD_LIBS_INIT})
if (TARGET LevelZero::ze_loader)
    target_link_libraries(${TARGET} PRIVATE LevelZero::ze_loader)
else()
    find_library(LEVEL_ZERO_LOADER ze_loader)
    if (LEVEL_ZERO_LOADER)
        target_link_libraries(${TARGET} PRIVATE ${LEVEL_ZERO_LOADER})
    endif()
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
