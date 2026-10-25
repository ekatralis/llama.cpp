# PTQ1_0 prompt processing on ROCm/HIP is much less accurate than generation (fp16 hipBLAS fallback)

## Summary

On ROCm (HIP) builds, PTQ1_0 matmuls with more than 8 tokens per batch, i.e. prompt processing, give logits
that are about **30x further from the CPU reference** than the same model's single-token generation path.
Measured as mean KL divergence against CPU logits: **0.0035 at ubatch 512 vs 0.00012 at ubatch 1**.
PQ2_0 on the same GPU does not show the problem.

Cause: PTQ1_0 has no MMQ path on HIP (`ggml_cuda_should_use_mmq` only enables it for NVIDIA Turing+), so every
PTQ1_0 matmul with batch > 8 falls back to `ggml_cuda_mul_mat_cublas`. On AMD GPUs that fallback dequantizes the
weights to fp16 and runs an fp16 GEMM with fp16 compute and fp16 output (`prefer_f32_output` is only set for
Volta / RDNA4 / CDNA). The fork's own comment in `mmq.cu` already names this fallback as "the source of
PTQ1_0's extra error on CUDA"; on HIP it is still the default for every prompt.

Confirmed: forcing the fallback to fp32 with the existing `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` drops the KLD from
0.0035 to 0.000066, closer to CPU than even the generation path. Enabling the existing dp4a MMQ path for PTQ1_0 on
HIP fixes it too, and is also ~1.8x faster at pp512 (see "Fix").

## Environment

- GPU: AMD Radeon VII (gfx906, Vega 20, 16 GB HBM2), single GPU
- ROCm 7.14 (hipcc 7.14.60850), Ubuntu 24.04, AMD Threadripper 1950X
- llama.cpp: PrismML fork `https://github.com/PrismML-Eng/llama.cpp`, branch `prism` @ `97ea5aeee`
  ("cuda: opt-in exact fusion of the GDN decode conv chain ... (#383)")
- Build: `HIPCXX=/opt/rocm/lib/llvm/bin/clang cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON
  -DGPU_TARGETS=gfx906 -DLLAMA_BUILD_TESTS=ON` (plus ccache launchers), no source changes
- Models: `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.53 GiB) and `Ternary-Bonsai-2-27B-PQ2_0.gguf` (6.70 GiB)

## Measurements

`llama-perplexity --kl-divergence`, 4 chunks of 512 tokens, `-fa on`, all layers on the GPU. The base logits come
from a CPU-only run of the same build (`HIP_VISIBLE_DEVICES=-1`, `-ngl 0`, ubatch 512). Text: ~900 KB of plain
English technical documentation (any text works).

| model | path | ubatch | mean KLD vs CPU | max KLD | same top token | PPL(GPU)/PPL(CPU) |
|---|---|---:|---:|---:|---:|---:|
| PTQ1_0 | stock HIP, generation path (MMVQ) | 1 | 0.000124 | 0.0066 | 99.6% | 1.0002 +- 0.0007 |
| PTQ1_0 | stock HIP, prompt path (fp16 hipBLAS) | 512 | **0.003501** | 0.0458 | 95.7% | 1.0045 +- 0.0028 |
| PQ2_0 | stock HIP, generation path | 1 | 0.000168 | 0.0030 | 98.6% | 1.0010 +- 0.0007 |
| PQ2_0 | stock HIP, prompt path (MMQ) | 512 | 0.000185 | 0.0111 | 98.8% | 1.0011 +- 0.0007 |
| PTQ1_0 | stock HIP, `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | 512 | **0.000066** | 0.0031 | 99.9% | 1.0007 +- 0.0006 |
| PTQ1_0 | PTQ1_0 MMQ (dp4a tiles) enabled on gfx906, see "Fix" | 512 | 0.000162 | 0.0283 | 99.5% | 1.0012 +- 0.0007 |

For scale: on upstream llama.cpp the same ubatch 512 vs ubatch 1 comparison for a Qwen3.8 27B Q3_K_XL quant
gives KLD 0.0007, and for PQ2_0 here the two paths are within 0.00002 of each other. PTQ1_0 at ubatch 512 is the
outlier. The stock default run was done three times, with identical results each time (KLD 0.003501), so this is not noise.

Prompt speed (llama-bench `-p 512 -n 0 -fa 1 -r 3`, PTQ1_0): stock default 192.6 t/s, stock with
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` 161.1 t/s, PTQ1_0 MMQ on gfx906 345 t/s.

The practical effect is a small but systematic quality loss in everything the model reads (prompt,
system prompt, documents), while everything it generates token by token is fine. Speculative decoding verify
steps with more than 8 tokens take the same path.

`test-backend-ops test -b ROCm0 -p "pq2_0|ptq1_0"` passes on this build (MUL_MAT 420/420), so the op tests do not
catch this: the per-matmul error stays within the test tolerance; it adds up across the 64 layers.

## Steps to reproduce

```sh
M=Ternary-Bonsai-2-27B-PTQ1_0.gguf
T=some-text.txt   # any plain text, a few hundred KB

# 1. CPU reference logits
HIP_VISIBLE_DEVICES=-1 ./build/bin/llama-perplexity -m $M -f $T -c 512 -b 512 -ub 512 --chunks 4 \
    -ngl 0 -t 16 --kl-divergence-base ptq1_0_cpu.kld

# 2. GPU, prompt path (batch 512 -> hipBLAS fallback)
./build/bin/llama-perplexity -m $M -f $T -c 512 -b 512 -ub 512 --chunks 4 -fa on -ngl 99 \
    --kl-divergence-base ptq1_0_cpu.kld --kl-divergence          # Mean KLD ~0.0035

# 3. GPU, generation path (batch 1 -> MMVQ), for comparison
./build/bin/llama-perplexity -m $M -f $T -c 512 -b 512 -ub 1 --chunks 4 -fa on -ngl 99 \
    --kl-divergence-base ptq1_0_cpu.kld --kl-divergence          # Mean KLD ~0.00012

# 4. Same as 2 with the BLAS fallback forced to fp32
GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 ./build/bin/llama-perplexity -m $M -f $T -c 512 -b 512 -ub 512 --chunks 4 \
    -fa on -ngl 99 --kl-divergence-base ptq1_0_cpu.kld --kl-divergence
```

Repeat with the PQ2_0 file to see that it is not affected.

## Root cause (code references at `97ea5aeee`)

1. `ggml/src/ggml-cuda/mmq.cu:366` `ggml_cuda_should_use_mmq`: the `GGML_TYPE_PTQ1_0` case (line 375-376,
   `mmq_supported = turing_mma_available(cc)`) is inside `#if !defined(GGML_USE_HIP)`. On HIP, PTQ1_0 hits
   `default: mmq_supported = false`. The `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH` block (line ~426) is also CUDA only.
2. `ggml/src/ggml-cuda/ggml-cuda.cu:1926` `ggml_cuda_mul_mat`: MMVQ takes batches up to `MMVQ_MAX_BATCH_SIZE` (8)
   (line 1979). Above that, MMQ is skipped (line 1989) and the matmul goes to `ggml_cuda_mul_mat_cublas` (line 1993).
3. `ggml-cuda.cu:1630`: for a quantized src0, `compute_type = fast_fp16_hardware_available(cc) ? F16 : F32`, and
   `fast_fp16_hardware_available` is true for every AMD GPU (`common.cuh:321`). So the weights are dequantized to
   fp16, src1 (activations) is converted to fp16, and the GEMM runs with `CUBLAS_COMPUTE_16F`.
4. `ggml-cuda.cu:1517`: `prefer_f32_output` (fp32 compute and output) is only set for Volta, RDNA4 and CDNA. On
   GCN (gfx906), RDNA1-3 the GEMM accumulates and writes in fp16.

Affected, by reading the code: every HIP build for PTQ1_0 batches of 9 or more tokens. RDNA4 and CDNA use fp32
compute in the fallback, so their error should be smaller (activations are still rounded to fp16); not tested.
Measured only on gfx906. PQ2_0, Q1_0 and Q2_0 are in the HIP MMQ list and do not take this path.

## Fix

Two options, both measured on gfx906 (option 2 through the env var):

1. **Enable the PTQ1_0 dp4a MMQ path on HIP** (preferred). The dp4a tile loader (`ggml_cuda_mmq_load_tiles_ptq1_0`)
   already exists; only the MMA tile layouts are NVIDIA specific. Changes needed:
   - `mmq.cu`: drop the `#if !defined(GGML_USE_HIP)` guards around the PTQ1_0 case in
     `ggml_cuda_mul_mat_q_switch_type` and in `ggml_cuda_should_use_mmq` (on HIP enable it for archs with dp4a, we
     only enabled and tested gfx906), and around the `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH` block.
   - Generate / keep the `mmq-instance-ptq1_0.cu` template instance for HIP builds (`generate_cu_files.py`).
   - `mmq-load-tiles.cuh`: on HIP, use `__builtin_amdgcn_perm` (`v_perm_b32`) instead of the emulated
     `__byte_perm`, and a carry-free per-byte subtract instead of the emulated `__vsub4` for the trit unbias.
     The generic versions also work; these are just faster on AMD.
   - An MMQ config row for PTQ1_0 (we used J = 64, 512 threads, I = 64, occupancy 2 on gfx906, same as PQ2_0).

   Result on gfx906: KLD vs CPU at ubatch 512 0.0035 -> 0.00016 (the same level as the generation path; MMQ
   quantizes activations to q8_1, so it does not reach the fp32 GEMM's 0.000066), and pp512 **193 -> 345 t/s**.
   The 345 includes two earlier gfx906 MMQ commits on our branch (128-bit y loads, y read from VRAM); the hipBLAS
   fallback also spent ~0.7 ms per matmul dequantizing the weights to fp16. Our patch:
   `HIP : PTQ1_0 MMQ (dp4a tiles) on gfx906` (`58543948f` on our branch `gfx906-bonsai`); happy to send it as a PR
   or a patch file. It is gated to gfx906 (`cc == GGML_CUDA_CC_VEGA20`) because that is the only AMD GPU we could
   test; other AMD architectures would need their own MMQ config rows and testing.

2. **Minimal fix**: in `ggml_cuda_mul_mat_cublas`, use fp32 compute for PTQ1_0 on AMD (or set `prefer_f32_output`
   for it). Measured with the env var (which does this for every BLAS call): KLD 0.000066, but pp512 drops from
   193 to 161 t/s. Fixes the accuracy, not the speed.

Workaround for users today: `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (see the table for its accuracy and speed cost).
