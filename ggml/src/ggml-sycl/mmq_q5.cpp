// mmq_q5.cpp - Q5 Quantization Launch Functions (Q5_0, Q5_1, Q5_K)
// Split from mmq.cpp refactoring

#include "mmq_internal.hpp"
#include "vecdotq.hpp"

void ggml_mul_mat_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_0_acc_ct1),
                            get_pointer(tile_x_d_q5_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:25: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_0_acc_ct1),
                            get_pointer(tile_x_d_q5_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_mul_mat_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_1_acc_ct1),
                            get_pointer(tile_x_dm_q5_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:27: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_1_acc_ct1),
                            get_pointer(tile_x_dm_q5_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_mul_mat_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_K_acc_ct1),
                            get_pointer(tile_x_dm_q5_K_acc_ct1),
                            get_pointer(tile_x_sc_q5_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:37: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_K_acc_ct1),
                            get_pointer(tile_x_dm_q5_K_acc_ct1),
                            get_pointer(tile_x_sc_q5_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

