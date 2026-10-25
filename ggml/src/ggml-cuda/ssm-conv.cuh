#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// fused GDN decode conv chain (1-8 tokens, one sequence); see ggml_cuda_gdn_conv_fusion in ggml-cuda.cu
#define GGML_CUDA_GDN_CONV_MAX_TOKENS 8
#define GGML_CUDA_GDN_CONV_MAX_DST    8
struct ggml_cuda_ssm_conv_gdn_fused_args {
    const float *   cache      = nullptr; // conv-state cache, rows of row_stride floats
    const int32_t * ids        = nullptr; // ids[0]: the row to read
    int64_t         row_stride = 0;
    const float *   x          = nullptr; // new tokens, token t of channel c at x[c*x_stride + t*x_tstride]
    int64_t         x_stride   = 0;
    int64_t         x_tstride  = 0;
    const float *   w          = nullptr; // conv weights, channel c at w[c*w_stride + j]
    int             w_stride   = 0;
    int             n_dst      = 0;       // conv-state rows written back (rollback snapshots: one per slot)
    float *         cache_dst[GGML_CUDA_GDN_CONV_MAX_DST] = {}; // [n_channels*(d_conv - 1)] each
    int             dst_off[GGML_CUDA_GDN_CONV_MAX_DST]   = {}; // first window element of each, 0..n_tokens
    float *         y          = nullptr; // silu(conv), [n_tokens][n_channels]
    float *         l2         = nullptr; // l2-normalised first n_l2 channels of each token, rows of row_len
    int             n_l2       = 0;
    int             n_channels = 0;
    int             n_tokens   = 0;
    int             row_len    = 0;
    int             d_conv     = 0;
    float           eps        = 0.0f;
};

void ggml_cuda_ssm_conv_gdn_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_gdn_fused_args & args);
