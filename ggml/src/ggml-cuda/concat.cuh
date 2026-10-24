#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Recurrent conv state shift (delta net / mamba style): conv_input = concat(state [dc, nch], x [n, nch]) and
// state_dst = last dc columns of conv_input. state is read from rows state_idx of state_src when state_idx != nullptr.
void ggml_cuda_op_concat_conv_state(ggml_backend_cuda_context & ctx, ggml_tensor * concat, ggml_tensor * cpy,
                                    const float * state_src, const int32_t * state_idx);
