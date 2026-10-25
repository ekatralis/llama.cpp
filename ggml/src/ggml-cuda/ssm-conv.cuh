#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// one token: ssm_conv + silu, then rms_norm + scale over the 128-wide q heads (first n_l2 channels) and k heads (next n_l2)
void ggml_cuda_op_ssm_conv_silu_l2(ggml_backend_cuda_context & ctx, const ggml_tensor * conv, ggml_tensor * silu,
    ggml_tensor * q_out, ggml_tensor * k_out, int n_l2, float eps, float scale_out);
