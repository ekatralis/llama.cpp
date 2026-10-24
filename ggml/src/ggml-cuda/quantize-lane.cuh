#pragma once

#include "common.cuh"

// In-kernel q8_1 quantization for producers of a MMVQ src1, same result as quantize_q8_1.

// One value per thread; 32 consecutive threads (32-aligned) form one block, i_cont is the flat index.
static __device__ __forceinline__ void quantize_q8_1_lane(const float xi, block_q8_1 * y, const int64_t i_cont) {
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    const int64_t ib  = i_cont / QK8_1;
    const int     iqs = i_cont % QK8_1;

    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(d, sum);
    }
}
