#include "common.cuh"

#if defined(__gfx906__)
// value of lane (lane ^ h) of the wave: DPP for h = 1, 2, 4, 8 (VALU, no LDS), ds_swizzle for 16, ds_bpermute for 32.
// h is a compile-time constant after unrolling, so the switch folds to one path. For 4 and 8 both shifts are taken by
// all lanes before the select: a DPP move inside a divergent branch reads 0 from the disabled lanes.
static __device__ __forceinline__ int ggml_cuda_gfx906_xor_lane(const int v, const int h) {
    switch (h) {
        case 1:  return __builtin_amdgcn_mov_dpp(v, 0xB1, 0xF, 0xF, true); // quad_perm [1, 0, 3, 2]
        case 2:  return __builtin_amdgcn_mov_dpp(v, 0x4E, 0xF, 0xF, true); // quad_perm [2, 3, 0, 1]
        case 4: {
            const int up = __builtin_amdgcn_mov_dpp(v, 0x104, 0xF, 0xF, true); // row_shl:4, lane i <- i + 4
            const int dn = __builtin_amdgcn_mov_dpp(v, 0x114, 0xF, 0xF, true); // row_shr:4, lane i <- i - 4
            return (threadIdx.x & 4) ? dn : up;
        }
        case 8: {
            const int up = __builtin_amdgcn_mov_dpp(v, 0x108, 0xF, 0xF, true); // row_shl:8
            const int dn = __builtin_amdgcn_mov_dpp(v, 0x118, 0xF, 0xF, true); // row_shr:8
            return (threadIdx.x & 8) ? dn : up;
        }
        case 16: return __builtin_amdgcn_ds_swizzle(v, 0x401F);              // bitmask mode, xor 16
        default: return __shfl_xor(v, h, 64);
    }
}

// 32-lane reductions (within each half of the wave) with DPP: pairs, quads, row_half_mirror (8), row_mirror (16),
// then one ds_swizzle (xor 16). Max and integer sums do not depend on the order, so they equal the shuffle versions.
static __device__ __forceinline__ float ggml_cuda_gfx906_reduce_max32(float x) {
    x = fmaxf(x, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0xB1,  0xF, 0xF, true)));
    x = fmaxf(x, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x4E,  0xF, 0xF, true)));
    x = fmaxf(x, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x141, 0xF, 0xF, true)));
    x = fmaxf(x, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x140, 0xF, 0xF, true)));
    x = fmaxf(x, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), 0x401F)));
    return x;
}

static __device__ __forceinline__ int ggml_cuda_gfx906_reduce_sum32(int x) {
    x += __builtin_amdgcn_mov_dpp(x, 0xB1,  0xF, 0xF, true);
    x += __builtin_amdgcn_mov_dpp(x, 0x4E,  0xF, 0xF, true);
    x += __builtin_amdgcn_mov_dpp(x, 0x141, 0xF, 0xF, true);
    x += __builtin_amdgcn_mov_dpp(x, 0x140, 0xF, 0xF, true);
    x += __builtin_amdgcn_ds_swizzle(x, 0x401F);
    return x;
}
#endif // defined(__gfx906__)

// In-register block FWHT over N floats held as NE = N/NT per thread, element i*NT + tid in reg[i].
// Shared by the standalone kernel (fwht_cuda_block) and the fused FWHT + q8_1 quantizer so both
// produce bit-identical transforms. `s` must hold N floats.
template <int N, int NT>
__device__ __forceinline__ void ggml_cuda_fwht_block_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
#if defined(__gfx906__)
            const float val2 = __int_as_float(ggml_cuda_gfx906_xor_lane(__float_as_int(val), h));
#else
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
#endif // defined(__gfx906__)
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);
