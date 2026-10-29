# Ternary Bonsai 2 27B on gfx906 (Radeon VII) - work log

Goal: see whether the PrismML Ternary Bonsai 2 27B models let a Qwen3.8 27B class model run with a much larger
context on a 16 GB Radeon VII, measure the stock PrismML fork, then port the gfx906 work of our own fork
(`/opt/llama.cpp`, branch `gfx906-qwen38`, documented in its `docs/gfx906/README.md`) and measure again.

- Repo: `/opt/llama.cpp-bonsai`, origin `https://github.com/PrismML-Eng/llama.cpp.git`
- Stock: branch `prism` @ `97ea5aeee` (upstream base is about 2026-08-25, two months older than our fork's `f5027fb`)
- Port: branch `gfx906-bonsai` (from `prism`)
- Git identity (repo-local only): Evangelos Katralis <vkatralis@gmail.com>. Commits carry an `Assisted-by: Claude`
  trailer, nothing is pushed.
- Remote `gfx906` = `/opt/llama.cpp` (read only). That clone is shallow, so `git fetch --update-shallow gfx906
  gfx906-qwen38` was needed (it records `f5027fb` as a shallow root in this repo).

## Hardware / software

Same machine as our fork: Radeon VII (gfx906, 60 CUs, 16 GB HBM2, ~850 GB/s measured streaming read),
Threadripper 1950X, Ubuntu 24.04, ROCm 7.14 (`/opt/rocm`). No packages were installed for this work.
GPU is shared with another agent: every GPU command runs under `/opt/gfx906-tools/gpu.sh` (flock).

## Models

| file | type | bpw | size | tensors |
|---|---|---:|---:|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | PTQ1_0 (ternary, group 128) | 1.75 | 5.53 GiB | 402 PTQ1_0, 96 BF16 (ssm_alpha / ssm_beta), F32 norms |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | PQ2_0 (2-bit, group 128) | 2.125 | 6.70 GiB | 402 PQ2_0, same BF16 / F32 |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` (reference) | mostly IQ4_XS / IQ3_S | ~3.9 | 12.2 GiB | |

Both Bonsai files: arch `qwen35`, 64 layers (48 gated delta net + 16 full attention, 24 Q / 4 KV heads, head dim 256),
context 262144, **no MTP (nextn) layer** (Qwen3.8 has 65 layers, the last one MTP). Every weight, including
`token_embd` and `output`, is PTQ1_0 / PQ2_0. The weights are "Hadamard folded" (`prism.hadamard.*`): the graph
applies a 1024-wide Walsh-Hadamard transform (with signs) to each matmul input (`MUL_MAT` with the hint
`GGML_HINT_SRC0_IS_HADAMARD`, run as `fwht_cuda_block`), and the inverse after the `token_embd` lookup.

Block formats (`ggml-common.h`):
- `block_ptq1_0`: 24 B of base-3 packed trits (5 per byte, 120 values) + 2 B (4 trits per byte, 8 values) + fp16 d =
  28 B per 128 weights. Weights are {-1, 0, 1} * d.
- `block_pq2_0`: fp16 d + 32 B of 2-bit codes = 34 B per 128 weights (only 2-byte aligned). Codes 0..3 -> {-1, 0, 1, 2}.

Weights read per generated token (all except `token_embd`): PTQ1_0 ~5.66 GB, PQ2_0 ~6.8 GB, Q3_K_XL ~12.6 GB.
At ~850 GB/s that is a floor of ~6.7 / ~8.0 / ~14.8 ms per token.

## Scripts

- `/opt/build-bonsai.sh`: copy of `/opt/build-gfx906.sh` for this repo. `BUILD_DIR=build-stock` (branch `prism`) or
  `BUILD_DIR=build-gfx906` (branch `gfx906-bonsai`); ninja + ccache, tests on, `-DGPU_TARGETS=gfx906`.
- `/opt/gfx906-tools/bonsai/` (the originals in `/opt/gfx906-tools` are unchanged):

| file | use |
|---|---|
| `bench.sh <build> <model> "<depths>" [reps]` | `llama-bench -fa 1 -p 512 -n 128 -d <depth>`, one gpu.sh lock per depth |
| `tgprof.py <build> <PTQ1_0\|PQ2_0> [ENV=VAL ...]` | rocprofv3 kernel trace of a 32-token generation: GPU ms/token, matvec ms/token, per matvec kernel / grid (`TOP=n` lists all kernels) |
| `pmc.sh <build> <model> [ENV=VAL ...]` | VALUBusy / MemUnitBusy / TA counters of the matvec kernels in a short generation |
| `kld-check.sh <PTQ1_0\|PQ2_0> [ubatch] [chunks]` | llama-perplexity KL divergence of a build vs the stock build (env `FORK`, `BASE_BIN`, `BASE_TAG`) |
| `maxctx.sh <bin dir> <model> <kv type> <lo> <hi>` | largest `-c` llama-server starts with (`-fit off`) and answers a request at |
| `ppprof.py <build> <model>` | rocprofv3 trace of one pp512 pass: GPU ms per pass by kernel |
| `ncprof.py <build> <gguf> <N>` | GPU / matvec ms per forward pass of N tokens (`llama-bench -p N`), for the MTP verify width |
| `mtp-test.sh <label> <bin dir> <gguf> [args]` | our mtp-test.sh for any binary / model (port 8199, `ignore_eos`, see MTP section) |
| `mmvq-bench.sh`, `kres.sh`, `kasm.sh` | adapted copies (paths) of our tools; `ops/bonsai_*.txt` = `test-export-graph-ops` of each model |

Note: `test-backend-ops perf` is not usable for kernel timing on this fork: every run has ~140 us of host gaps
(a 25 us kernel reports 180 us/run). Kernel times below come from rocprofv3 traces of real generation.

## Step 1: stock build

`BUILD_DIR=build-stock /opt/build-bonsai.sh` on `prism` @ `97ea5aeee`: builds cleanly for gfx906, no fix needed.
PTQ1_0 and PQ2_0 have HIP paths: on HIP they go through the generic `mul_mat_vec_q` kernel with the plain q8_1 (AoS)
activation layout (the PT / SoA / PQ2 layouts and the dedicated PTQ1_0 / PQ2_0 kernels are CUDA-only), MMQ for
prompts. The FWHT+quantize fusion and the GDN fusions are gated to NVIDIA.

## Step 2: correctness of the stock fork

`test-backend-ops test -b ROCm0 -p "pq2_0|ptq1_0"`: MUL_MAT 420/420, GET_ROWS 12/12, MUL_MAT_ID 166/166 pass.

Model op files (`--test-file ops/bonsai_*.txt`): all 16 PTQ1_0 / PQ2_0 matmul cases pass. Failures that are test
artifacts, not kernel bugs:
- `MUL_MAT f32[1024,1024] x [1024,n]` with `op_params [1:1]` (the Hadamard hint): ERR ~342. The hint makes a
  backend run the transform instead of multiplying with src0, and the test fills src0 with random data.
- SCALE / CPY / GET_ROWS / ROPE / SSM_CONV with NaN in both backends (random recurrent state inputs), as in our fork.
- The full file run segfaults (host side) in `ggml_cuda_op_gated_delta_net_impl`: the fork's GATED_DELTA_NET reads
  `src[7]` / `src[8]` (raw gates), which the exported test case does not have.

KL divergence vs CPU logits (`HIP_VISIBLE_DEVICES=-1 llama-perplexity ... -ngl 0 -t 16`, kld-text.txt, c 512,
4 chunks; CPU PPL 6.0356 for PTQ1_0, 6.0364 for PQ2_0):

| comparison | mean KLD | max KLD | same top |
|---|---:|---:|---:|
| PTQ1_0 stock ROCm ub 1 vs CPU | 0.000124 | 0.0066 | 99.6% |
| PTQ1_0 stock ROCm ub 512 vs CPU | 0.003501 | 0.0458 | 95.7% |
| PTQ1_0 stock ROCm ub 512 vs stock ROCm ub 1 | 0.003525 | 0.0485 | 95.5% |
| PQ2_0 stock ROCm ub 1 vs CPU | 0.000168 | 0.0030 | 98.6% |

The stock prompt path (ub 512: MMQ, FWHT on 512 columns, chunked GDN, FA) is ~30x further from CPU than the
generation path; for Qwen3.8 Q3_K_XL on upstream the same ub 512 vs ub 1 gap is 0.0007. PPL ratio 1.0045 +- 0.0028,
so small but a stock-fork issue. Cause: on HIP PTQ1_0 prompts run fp16 dequantize + rocBLAS with fp16 activations
(no PTQ1_0 MMQ on HIP); fixed on gfx906 by `58543948f`, see "Prompt processing".

## Step 3: stock performance

`bench.sh build-stock <model> "0 8192" 3` and `"32768 65536" 1` (llama-bench `-fa 1 -p 512 -n 128`, f16 KV):

| model | pp512 | tg128 | pp512 @ d8192 | tg128 @ d8192 | pp512 @ d32768 | tg128 @ d32768 | pp512 @ d65536 | tg128 @ d65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PTQ1_0 | 193.19 +- 5.53 | 22.74 +- 0.08 | 181.43 +- 4.88 | 21.25 +- 1.17 | 155.32 | 19.73 | 128.61 | 17.45 |
| PQ2_0 | 141.01 +- 4.40 | 33.75 +- 0.13 | 136.24 +- 1.83 | 31.99 +- 0.26 | 121.09 | 27.23 | 104.04 | 22.89 |

(d32768 / d65536: one repetition each.) For reference, Qwen3.8 Q3_K_XL: upstream 241 / 22.2, our fork 386 / 36.1.

## Step 4: where the time goes (stock, generation)

`tgprof.py build-stock PTQ1_0` (32 tokens): 43.6 ms GPU per token, matvec (`mul_mat_vec_q`) 35.9 ms = 82%,
~158 GB/s on 5.66 GB. PQ2_0: 29.4 ms GPU per token, matvec 22.0 ms (~310 GB/s).
Rest per token: `quantize_q8_1` 361 launches 1.36 ms, `fwht_cuda_block` 257 launches 1.2 ms, BF16 alpha / beta
`mul_mat_vec_f` 96 launches 1.03 ms, `rms_norm` 0.98 ms, gated delta net 0.96 ms, `cpy_scalar` 0.57 ms, FA 0.41 ms.

Why the stock matvec is slow on gfx906:
- PTQ1_0: `VDR = QI = 4`, one thread decodes a whole 128-weight block, and reads its 128 activations as 36 scattered
  4-byte loads out of 36-byte `block_q8_1` structs (144 B of activations per 28 B of weights). With the GCN table
  (2 warps x 64 lanes) a K = 5120 row (40 blocks) keeps 40 of 128 threads busy, K = 17408 (136 blocks) 53%.
- PQ2_0: 4 lanes per block, 8 x 2-byte loads per lane (blocks are 2-byte aligned), 2 warps per row.

## Step 5: port (branch gfx906-bonsai)

| commit | change | effect |
|---|---|---|
| `e10d40895` | our `a25d385`: MMVQ table `MMVQ_PARAMETERS_VEGA20`, batch 1 = 1 warp x 2 rows (gfx906 only). Re-done by hand (conflicts with the fork's PTQ1_0 table code) | matvec ms/token PTQ1_0 35.9 -> 23.7, PQ2_0 22.0 -> 15.7 |
| `d5dcc70a2` | new gfx906 one-column PTQ1_0 / PQ2_0 kernel `mul_mat_vec_q_lg` on the PT / PQ2 activation layouts (below) | matvec PTQ1_0 23.7 -> 13.1, PQ2_0 15.7 -> 12.7 ms/token; tg128 22.7 -> 48.4 / 33.8 -> 50.2 |

`mul_mat_vec_q_lg` (mmvq.cu, gfx906 only, one column, no ids, plain 2D; off with `GGML_CUDA_NO_MMVQ_HW=1`):
- `ggml_cuda_q8_1_layout_host` returns the fork's existing PT layout (PTQ1_0) or PQ2 layout (PQ2_0) on gfx906, so the
  quantizer writes activations with exact int16 block sums and 16-byte aligned pieces. The digit bias (codes - 1) is
  one integer subtraction per 32 values instead of a decode per weight.
- A group of `lpr` lanes owns `R` rows; each lane strides whole x blocks along K and reuses each activation piece for
  all rows of the group. Groups of 8 / 32 lanes make K = 5120 / 6144 / 17408 divide evenly (no idle tail).
- `amdgpu_waves_per_eu` + `amdgpu_flat_work_group_size` instead of `__launch_bounds__` (README finding).
- PTQ1_0: one 128-weight block per lane (`ptq1_0_pt_block_dot`, made HIP-ready: `__byte_perm` -> one `v_perm_b32`,
  the digit dot -> signed `v_dot4`, exact since digits are 0..2), 1 wave per block, 8 lanes per row, 1 row,
  2 waves per EU (up to 128 VGPRs, no spills).
- PQ2_0: one 32-weight chunk per lane (shift + mask per 4 codes, no lookup), 1 wave, 32 lanes per row, 2 rows,
  4 waves per EU.
- Fused gate (ffn gate + up + SWIGLU) and bias epilogues as in the generic kernel.

Launch sweep (matvec ms per generated token, `tgprof.py`, temporary knob since removed):

| PTQ1_0 variant | waves x lanes/row x rows, waves/EU | ms |
|---|---|---:|
| generic vec_dot in half-wave rows (our cf35d3b shape) | 4 x 32 x 2, 4 | 46.6 (spills: 128-weight decode per lane) |
| PT layout, whole block per lane | 4 x 8 x 1, 4 | 46.6 (spills) |
| PT, block split over 2 lanes, both lanes decode words 4/5 | 4 x 16 x 1, 4 | 16.5 |
| PT, block split over 2 lanes, no duplicate decode | 4 x 16 x 1, 4 | 21.8 |
| PT, whole block per lane | 4 x 8 x 1, 2 | 13.4 |
| PT, whole block per lane (kept) | 1 x 8 x 1, 2 | 13.1 |

The "no duplicate decode" split has fewer VALU instructions (182 vs 205 per loop) but is slower: counters show
MemUnitBusy 82-90% / VALUBusy 33-56% vs 72-80% / 53-69%. With the two-lane split each lane needs 12-15 load
instructions per 64 weights; the memory unit (address processing per load instruction) is the limit, not the ALU.
One lane per block needs 11 loads per 128 weights (16-byte PT pieces) but ~80-115 VGPRs, hence 2 waves per EU.

PQ2_0 (ms): 4 x 32 x 2 at 4/EU 12.8; 1 x 32 x 2 at 4/EU 12.66 (kept); 32 x 3 / 32 x 4 / 64 x 2 / 8 x 1 slower.

Per call after `d5dcc70a2` (generation, 1 wave per block): PTQ1_0 ffn gate+up fused 66 us (590 GB/s), ffn 17408 rows
40 us (490 GB/s), output head 483 us (575 GB/s); PQ2_0 ffn_down 30 us (~800 GB/s), gate+up 70 us (670 GB/s),
output head 470 us (720 GB/s).

Correctness after `d5dcc70a2`: MUL_MAT 420/420, MUL_MAT_VEC_FUSION 74/74 (incl. the fork's large-K gate cases), model
matmul shapes 16/16 per model. KLD (generation, ub 1):

| comparison | mean KLD | max KLD | same top |
|---|---:|---:|---:|
| PTQ1_0 gfx906 vs stock (2 chunks) | 0.000112 | 0.0016 | 99.2% |
| PQ2_0 gfx906 vs stock (2 chunks) | 0.000101 | 0.0010 | 98.6% |
| PTQ1_0 gfx906 vs CPU (4 chunks) | 0.000116 | 0.0034 | 99.4% (stock: 0.000124) |
| PQ2_0 gfx906 vs CPU (4 chunks) | 0.000159 | 0.0029 | 99.0% (stock: 0.000168) |

Not bit-identical to stock (summation order), as close to the CPU reference as stock.

### Generation: more fusions

| commit | change | GPU ms per token (PTQ1_0 / PQ2_0, tgprof) |
|---|---|---:|
| (after `d5dcc70a2`) | | 20.3 / 20.1 |
| `05f271e2b` | the fork's fused Hadamard transform + q8_1 quantize (`ggml_cuda_try_fwht_q8`, NVIDIA only) enabled on gfx906; the fused quantizer learned the PQ2 layout. Removes the 361 quantize launches per token | 20.0 / 19.3 |
| `2a6fa4968` | the fork's opt-in GDN fusions (conv chain: gather + concat + state copy + ssm_conv + silu + q/k L2 in one launch; alpha/beta BF16 mat-vec pair) enabled and on by default on gfx906 (`GGML_CUDA_GDN_FUSION=0` turns them off) | 18.7 / 17.9 |

Both are bit-identical to the unfused path (KLD at noise level, 100% same top token). Generation profile now
(PTQ1_0 / PQ2_0, ms per token): matvec 12.6 / 12.1, fused transform + quantize 2.0 / 1.8 (257 launches, 5 workgroups
each for K = 5120, latency bound: 24 + ~60 `ds_bpermute` shuffles per thread on HIP), rms_norm 0.9, gated delta net
0.9, alpha/beta pair 0.56, flash attention 0.36.

## Prompt processing

Stock profile, one pp512 pass (`ppprof.py`):
- PTQ1_0 has no MMQ path on HIP (`mmq_supported` is CUDA only): every matmul dequantizes PTQ1_0 to fp16 (0.7 ms each)
  and runs a rocBLAS fp16 GEMM (2-9 ms each, ~14 TFLOPS). The fork's own comment: the fp16 dequant + BLAS path is
  "the source of PTQ1_0's extra error" (activations rounded to fp16) - the ubatch 512 KLD of step 2.
- PQ2_0 runs MMQ (dp4a, q8_0 vec dot) with the RDNA2 config rows (this fork has no GCN table): 8.1 ms per matmul,
  ~8 TOPS, 90% of the pass.
- Gated delta net 4.2 ms per layer (205 ms per pass), FA 11.5 ms, concat 3.4 ms (the fork's transposing concat), BF16
  alpha/beta GEMMs (rocBLAS) 34 ms per pass.

| commit | change | pp512 PTQ1_0 / PQ2_0 |
|---|---|---:|
| stock | | 193 / 141 |
| `387547f3e` | our `34b532d`: 128-bit LDS y reads in the q8_0 dp4a vec dot (cherry-pick, clean) | - |
| `415633332` | our `6e5f055`: MMQ y read from VRAM with scalar loads (`y_vram`, needs <= 8 columns per warp); a small `mmq-config-gfx906.cuh` (PQ2_0 / PTQ1_0 rows, RDNA2 table for every other type) with 512 threads, I = 64, occupancy 2 at J = 64. Logits identical to stock | 193 / 346 |
| `58543948f` | PTQ1_0 MMQ enabled on gfx906 (the fork's dp4a tile loader; MMA layouts stay NVIDIA only), loader on HIP with `v_perm_b32` and a carry-free unbias instead of the emulated `__byte_perm` / `__vsub4` | 345 / 345 |
| `41664562b` | gated delta net: on gfx906 a wave holds 16 state columns of 4 lanes (upstream later moved to 4 columns per warp); per-token reductions are 4-lane shuffles. 4.29 -> 1.07 ms per layer | 384 / 383 (profile pass), llama-bench 407 / 405 |

MMQ sweep at J = 64 (PQ2_0, ms per pass): 512 thr / I 64 / occ 2: 1034 (kept); I 128: 1234; occupancy 1: same.
MMQ now runs at ~24 TOPS on every large shape (~45% of the dp4a peak), 84% of the pass; the 1024-row K/V shapes
16 TOPS (10.7 ms per pass, few tiles: the GCN J rebalance would save ~3 ms, not ported). GDN lane segments: 4 per
wave 1.69 ms, 8: 1.46 ms, 16: 1.07 ms (kept), 32: wrong for S_v = 64 and spills.

Prompt-path accuracy (KLD vs CPU logits, ubatch 512, 4 chunks): PTQ1_0 stock 0.003501 (max 0.046) -> MMQ 0.000162 ->
with the GDN change 0.000116 (max 0.0036); PQ2_0 stock 0.000185 (max 0.011) -> 0.000165 (max 0.0038).

Tried and dropped:
- Our GDN k prefetch (`9a410ac`): on this fork's (old layout) kernel 4.27 -> 4.45 ms, slower; not needed after the
  segment change.
- Our GCN MMQ table (`mmq-config-gcn.cuh`, upstream #27841): not brought over. The Bonsai types only need their own
  rows (the small gfx906 table), and the other commits that edit it tune IQ / K-quant rows these models do not use.

At depth 32768 a pp512 pass is MMQ 66% and flash attention (`flash_attn_tile<256, 256, 16, 2>`, 21.5 ms per call)
22%: FA is what grows with depth.

## MTP: community graft decent-jawfish/bonsai-2-27b-mtp

Hugging Face repo `decent-jawfish/bonsai-2-27b-mtp` @ `5edf5f552d45`, downloaded to `/opt/llms/bonsai-mtp/`
(`Bonsai-2-27B-PQ2_0-MTP.gguf`, sha256 `78df4279...dbd847` = the LFS oid; README.md, graft_mtp.py, the patch).
Read before use:
- `graft_mtp.py`: a plain GGUF rewrite. It range-downloads the 15 `blk.64.*` tensors (the MTP / nextn block) of
  `unsloth/Qwen3.8-27B-GGUF` UD-Q2_K_XL, appends them to the Bonsai PQ2_0 file, sets `block_count` 65 and
  `nextn_predict_layers` 1. The MTP block is not Hadamard folded (the README argues the residual stream is in the
  original basis: Bonsai's norm weights match Qwen3.8). Not run here (the merged GGUF is provided).
- `0001-qwen35-mtp-hadamard-inverse.patch`: applies the Hadamard inverse after the MTP graph's own token-embedding
  lookup in `src/models/qwen35.cpp` (the trunk does this in `build_inp_embd`; without it
  `llama_verify_hadamard_graph` refuses the context). `git apply` reports it corrupt (truncated hunk), and it is not
  needed: `prism` already has the same change (`518ad108f`, "qwen35: apply the Hadamard inverse to the MTP
  token-embedding lookup"). No branch was needed for it; the stock fork runs the model as is.

Checks: all 851 base tensors are byte-identical to `Ternary-Bonsai-2-27B-PQ2_0.gguf`; added: 15 tensors (Q6_K / Q8_0
weights, F32 norms, 0.35 GB). Main-model logits: KLD vs CPU at ubatch 512 is exactly the plain PQ2_0 number (0.000185,
max 0.011131), i.e. identical.

`mtp-test.sh` (llama-server, `-c 8192`, greedy, 256 tokens, the two prompts of our `/opt/gfx906-tools/mtp-test.sh`).
With the raw CSV prompt Bonsai's first greedy token is EOS, so the Bonsai script sends `ignore_eos: true`.
t/s for the two prompts, accepted / drafted tokens:

| build | no MTP | draft 2 | draft 3 | draft 4 |
|---|---:|---:|---:|---:|
| stock `prism` | 33.4 / 33.2 | 42.9 / 40.7 (140/229, 134/239) | 40.2 / 38.9 (153/303, 150/310) | - |
| `58543948f` (1-column kernel only) | 52.2 / 52.5 | 44.4 / 42.0 | 41.9 / 40.6 | - |
| `ee5c849aa` (PQ2_0 kernel for 2-4 columns) | 52.2 / 52.5 | 60.0 / 57.0 | 54.3 / 52.4 | 41.5 / 37.5 |
| `44cc405ce` (2-4 column launch shapes) | 51.7 / 51.8 | **64.6 / 60.9** | 60.8 / 58.3 | |
| reference: Qwen3.8 Q3_K_XL on our fork | 36.2 / 36.3 | 58.8 / 55.2 | 56.6 / 51.4 | |

Acceptance is the same on every build (61% / 56% at draft 2, the README reports 0.60). Draft 4 verifies 5 columns,
which is outside the lane-group kernel.

- `ee5c849aa`: the lane-group kernel takes 1-4 columns for PQ2_0 (PQ2 layout holds several columns), one weight
  decode shared by the columns. Before, the 3-column verify ran the generic kernel and MTP was slower than no MTP.
- `44cc405ce`: rows per 32-lane group / waves per EU per column count (`ncprof.py`, matvec ms per pass):
  2 columns 16.2 -> 14.9 (4 rows, 3/EU), 3 columns 20.3 -> 18.3 (3 rows, 3/EU), 4 columns 26.7 -> 22.5 (4 rows, 2/EU).
- KLD vs stock at ubatch 4: 0.000106 (ubatch 1: 0.000101).

MTP server step profile (draft 2): the 3-column PQ2_0 mat-vec is 57% of GPU time (54 us per call vs ~27 us for one
column); the MTP head's Q6_K mat-vecs use the generic kernel; the 3-token GDN conv chain is not fused (the fork's
fusion is for one token).

## Max context in 16 GB

`maxctx.sh` (llama-server `-ngl 99 -fa on -fit off`, KV and compute buffers allocated at start, one request served),
4096 steps:

| model | build | f16 KV | q8_0 KV |
|---|---|---:|---:|
| PTQ1_0 | stock and gfx906-bonsai | 155,648 | **262,144** (full model context) |
| PQ2_0 | stock and gfx906-bonsai | 139,264 | 233,472 |
| PQ2_0 + MTP, `--spec-type draft-mtp --spec-draft-n-max 2` | gfx906-bonsai | 106,496 | 167,936 |
| Qwen3.8 Q3_K_XL | stock PrismML fork | 57,344 | 98,304 |
| Qwen3.8 Q3_K_XL | our gfx906 fork | 57,344 | 98,304 |

KV cache is ~64 KiB per token in f16 (16 attention layers x 4 KV heads x 256 x 2) and half that in q8_0; the recurrent
state is constant. The MTP draft context adds a second KV and compute buffers.

## Final results (llama-bench `-fa 1 -p 512 -n 128`, f16 KV; d0 / d8192: 3 reps, d32768 / d65536: 1 rep)

| model | build | pp512 | tg128 | pp512 @ d8192 | tg128 @ d8192 | pp512 @ d32768 | tg128 @ d32768 | pp512 @ d65536 | tg128 @ d65536 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PTQ1_0 | stock | 193.19 | 22.74 | 181.43 | 21.25 | 155.32 | 19.73 | 128.61 | 17.45 |
| PTQ1_0 | `41664562b` | 407.42 | 51.03 | 359.36 | 48.34 | 268.12 | 38.82 | 197.29 | 30.39 |
| PQ2_0 | stock | 141.01 | 33.75 | 136.24 | 31.99 | 121.09 | 27.23 | 104.04 | 22.89 |
| PQ2_0 | `41664562b` | 405.38 | 53.82 | 357.79 | 49.48 | 267.49 | 39.57 | 196.40 | 30.93 |
| Q3_K_XL (ref) | our fork | 386 | 36.1 | | 34.25 | | | | |

GPU time per generated token now: PTQ1_0 18.7 ms (matvec 12.6), PQ2_0 17.9 ms (matvec 12.1); the weight-read floor
at ~850 GB/s is ~6.7 / ~8.0 ms. PQ2_0 decodes faster than PTQ1_0 despite 20% more bytes: the base-3 trit decode costs
~3 VALU instructions per weight and keeps PTQ1_0's mat-vec at 450-590 GB/s, while PQ2_0 reaches 620-800 GB/s.

## Ported vs not ported (our gfx906-qwen38 commits)

| our commit | here |
|---|---|
| a25d385 MMVQ VEGA20 launch shape | ported (`e10d40895`, by hand) |
| cf35d3b / e64e5f4 half-wave MMVQ, amdgpu_waves_per_eu, split load / math | technique ported as the PTQ1_0 / PQ2_0 lane-group kernel (`d5dcc70a2`, `ee5c849aa`, `44cc405ce`); the IQ-specific code does not apply |
| 34b532d 128-bit MMQ y reads | ported (`387547f3e`) |
| 6e5f055 MMQ y from VRAM | ported (`415633332`) with a gfx906 table for the Bonsai types |
| 9a410ac GDN k prefetch | tried, slower on this kernel; replaced by the lane-segment change (`41664562b`) |
| 19194a1 tiled transposed concat | fork has an equivalent (#309) |
| d402b46 q8_1 reuse, 6cdfbae in-kernel quantize | covered by the fork's fused Hadamard transform + quantize (`05f271e2b`): every matmul input here is a Hadamard output |
| 18c3d74, cfc6d64, 052ca53, e192e1e, edb1def / e919904 (GDN gate / state / conv fusions) | the fork's own GDN fusions cover the one-token case (`2a6fa4968`); multi-token (MTP verify) conv fusion not done |
| 0b56028, 2180fea, 009bc7d (IQ grids / signs / split paths) | no IQ / K-quant weights in these models |
| 91861c5, 8e1d244, 3c1147a, bafd5d3, fc5e829, b99ad83 (GCN MMQ table, J rebalance, K-quant y_vram) | need upstream's mmq-config-gcn.cuh; the Bonsai rows are in the small gfx906 table instead; J rebalance worth ~3 ms per pp512 pass |
| 697ee81 / e7d324f ysh, 5fc43a2 narrow concat, 19d64c7 rms_norm q8_1, a565517 residual add in MMVQ | not needed / small for these models (activations go through the fused Hadamard quantizer) |

## Remaining bottlenecks

- Generation: PTQ1_0 mat-vec is decode / memory-instruction bound at ~450-590 GB/s (12.6 of 18.7 ms); the fused
  Hadamard transform + quantize (257 launches, ~2 ms per token) is latency bound by `ds_bpermute` shuffles (DPP
  reductions would be the next step).
- MTP: the 3-column verify mat-vec (57% of the step), unfused 3-token GDN conv chain.
- Prompt: MMQ at ~24 TOPS (84% of a pp512 pass); at depth the flash-attention tile kernel.

## Round 2: speed round (2026-10-25)

### Stock reference logits (reused by every check)

`/opt/gfx906-tools/bonsai/ref/<Q>_stock_ub<UB>_c4.kld` for Q in PTQ1_0, PQ2_0 and UB in 1, 3, 512 (SHA256SUMS next to
them), made by the stock build (`prism` @ `97ea5aeee`, `build-stock`) with

    /opt/gfx906-tools/gpu.sh /opt/llama.cpp-bonsai/build-stock/bin/llama-perplexity \
        -m /opt/llms/Ternary-Bonsai-2-27B-<Q>.gguf -f /opt/gfx906-tools/kld-text.txt \
        -c 512 -b 512 -ub <UB> --chunks 4 -fa on -ngl 99 --kl-divergence-base <Q>_stock_ub<UB>_c4.kld

(stock PPL over the 4 chunks: PTQ1_0 6.0342 / 6.0368 / 6.0603 at ub 1 / 3 / 512, PQ2_0 6.0397 / 6.0412 / 6.0404).
Check a build against them: `kld-ref.sh [build dir] ["PTQ1_0 PQ2_0"] ["1 3 512"]` (same command with
`--kl-divergence`, prints KLD / max / same top / PPL ratio per model and ubatch). CPU references (ubatch 512, 4 chunks):
`/opt/gfx906-tools/bonsai/kld/<Q>_cpu_ub512_c4.kld` (`HIP_VISIBLE_DEVICES=-1 ... -ngl 0 -t 16`).

KLD vs the stock references at the end of the round (`c82cd0052`):

| model | ub 1 | ub 3 | ub 512 |
|---|---:|---:|---:|
| PTQ1_0 | 0.000136 (max 0.0061, top 99.6%) | 0.000116 (top 99.1%) | 0.003767 (stock's own fp16 prompt-path error, see step 2) |
| PQ2_0 | 0.000111 (top 98.9%) | 0.000119 (top 99.3%) | 0.000155 (top 99.1%) |

### Commits

| commit | change | effect |
|---|---|---|
| `fd1a33bd2` | our a565517: residual add fused into the 2-4 column PQ2_0 lane-group mat-vec (`x_bias_stride_col`, `ggml_cuda_mmvq_fuses_bias_multi`), test MUL_MAT_ADD_RESIDUAL | 3-token pass 26.82 -> 26.48 ms; MTP d2 64.6 / 60.9 -> 65.5 / 61.0 |
| `a6e2b3f65` | the fork's alpha / beta mat-vec pair (one launch, second weight as channel 1) for 1-8 columns, test MUL_MAT_VEC_PAIR 2-4 columns | 3-token pass 26.48 -> 25.78 ms, bit-identical |
| `ee6874b5f` | the fork's GDN conv chain fusion (gather + concat + state copies + ssm_conv + silu + q/k L2) for 1-8 tokens and up to 8 rollback snapshot copies (with MTP every pass writes K copies, so it never matched while serving). Test GDN_CONV_CHAIN (n_t, K) up to (8, 3) + unfused 9 | 3-token pass 25.78 -> 25.24 ms; MTP d2 -> 68.6 / 64.5, d3 60.7 / 58.0 -> 63.4 / 60.5; bit-identical |
| `42245787e` | gfx906 DPP lane exchanges in the Hadamard butterfly (xor 1 / 2 quad_perm, 4 / 8 row shifts, 16 ds_swizzle) and DPP 32-lane max / int-sum in the fused quantizer | fused transform + quantize 6.9 -> 5.5 us (PQ2 layout), 7.8 -> 6.9 us (PT), 257 per token; bit-identical |
| `288c5a39a` | PQ2_0 lane-group mat-vec for 5 columns (4 rows / 32-lane group, 2 waves per EU) | MTP d4 41.5 / 37.5 -> 58.6 / 53.4 |
| `c82cd0052` | gfx906 tile FA: for 1-4 tokens and GQA 5-7 one tile of 8 Q heads per K/V head (padding columns skipped), so each K/V head is read once instead of 3 times; partial-tile indexing only in the HIP ncols2 = 8 instances. Tests: FLASH_ATTN_EXT head 256, GQA 6, 1-5 tokens; all 4355 FA cases pass | decode FA at a 33280 context 448 -> 232 us per layer; tg128 @ d8192 +6%, @ d32768 PQ2_0 39.57 -> 45.74 (+16%) |

### Final before / after

llama-bench `-fa 1 -p 512 -n 128`, f16 KV, 3 reps (d32768: 1 rep). Server: `mtp-test.sh` (llama-server `-c 8192`,
greedy, 256 tokens, the two prompts of our mtp-test.sh, `ignore_eos`), MTP rows with the PQ2_0 MTP graft
(`/opt/llms/bonsai-mtp/Bonsai-2-27B-PQ2_0-MTP.gguf`, `--spec-type draft-mtp --spec-draft-n-max N`); there is no PTQ1_0
MTP model.

| | stock `prism` | round 1 end (`7d018b4df`) | round 2 end (`c82cd0052`) |
|---|---:|---:|---:|
| PTQ1_0 pp512 | 193.19 | 407.42 | 407.94 |
| PTQ1_0 tg128 | 22.74 | 51.03 | 52.56 |
| PTQ1_0 pp512 @ d8192 | 181.43 | 359.36 | 359.01 |
| PTQ1_0 tg128 @ d8192 | 21.25 | 48.34 | 51.44 |
| PQ2_0 pp512 | 141.01 | 405.38 | 404.96 |
| PQ2_0 tg128 | 33.75 | 53.82 | 53.32 |
| PQ2_0 pp512 @ d8192 | 136.24 | 357.79 | 358.62 |
| PQ2_0 tg128 @ d8192 | 31.99 | 49.48 | 51.82 |
| PQ2_0 tg128 @ d32768 | 27.23 | 39.57 | 45.74 |
| server PTQ1_0, no MTP | - | - | 51.8 / 52.0 |
| server PQ2_0+MTP file, no MTP | 33.4 / 33.2 | 51.7 / 51.8 | 52.8 / 52.7 |
| server MTP draft 2 | 42.9 / 40.7 | 64.6 / 60.9 | **68.8 / 64.8** |
| server MTP draft 3 | 40.2 / 38.9 | 60.8 / 58.3 | 64.0 / 61.1 |
| server MTP draft 4 | - | 41.5 / 37.5 | 57.8 / 53.0 |

(Qwen3.8 Q3_K_XL on our fork, same test: no MTP 36.2 / 36.3, draft 2 58.8 / 55.2, draft 3 56.6 / 51.4.)
tg128 at d0 moves within run-to-run noise (+-0.4); the FA change shows at depth.

### Tried and dropped (round 2)

- PQ2_0 lane-group loop with per-lane pointer increments (no per-step 64-bit address math, ~20 of 210 loop instructions):
  matvec ms per pass n=1 11.87 -> 12.43, n=3 18.18 -> 21.17 (more live 64-bit pointers). Reverted.
- 5-column launch shapes (matvec ms per 5-token pass): 4 rows / 2 per EU 26.85 (kept), 3 / 2 28.09, 2 / 3 29.90, 2 / 2 31.34.
- PTQ1_0 decode through a 256-entry LDS table (byte -> 5 digits; 4 ds_read_b64 + 11 v_perm per 4-byte word instead of
  27 VALU): matvec 13.21 -> 14.12 ms per token (bank-conflicted random LDS reads; 128 VGPRs + 1 spill with the gate;
  the unused table code alone cost 12.68 -> 13.21). Reverted.
- PTQ1_0 repacked weight layout: not prototyped (see "Further improvements": loads are not the limit).
- Padded-GQA indexing compiled into every tile FA instance: 512-token FA at 33K 42.15 -> 48.04 ms (+14%); now only in
  the HIP ncols2 = 8 instances (42.15 again).
- rms_norm writing q8_1 (our 19d64c7): no target; every trunk norm output feeds the fused Hadamard quantizer or the
  BF16 alpha / beta mat-vecs. Only the MTP head quantizes normally.
- ffn gate + up + SWIGLU fusion is refused in 24 of 64 layers at one token: ggml-alloc gives the GLU output the buffer of
  the Hadamard output, which holds the fused quantizer's q8 rows, so `ggml_cuda_check_fusion_memory_ranges` fails
  (correctly). Not fixed (see further improvements).

### Open items (handed to the main session)

| item | estimate | notes |
|---|---|---|
| PTQ1_0 mat-vec (decode bound) | <= 10% of its 12.6 ms | see below; a load-time repack to 2-bit codes = PQ2_0 format, no gain over the PQ2_0 model |
| GCN J rebalance for the 1024-row K/V MMQ shapes | ~3 ms per pp512 pass (0.25%) | our 3c1147a / bafd5d3; the shapes run at 16 TOPS vs 24 elsewhere |
| fewer Hadamard launches | ~0.45 ms per token (2.4%) | fold rms_norm (x weight) into the fused transform for the 80 inputs whose only consumer is the transform |
| more multi-column tuning | 5-10% of a verify pass | 3-column mat-vec is 68% of a 3-token pass |

## Further improvements

Costs from rocprofv3 traces at `c82cd0052` (`tgprof.py` = 32-token generation, `ncprof.py` = N-token pass,
`ppprof.py` = pp512 pass). GPU time per token at d0: PTQ1_0 18.7 ms, PQ2_0 17.8 ms; 3-token pass (PQ2_0) 24.8 ms;
pp512 pass ~1230 ms.

1. **PTQ1_0 mat-vec** - 12.6 ms per token (67% of PTQ1_0 generation). Kernel `mul_mat_vec_q_lg<PTQ1_0, 1, 1, 8, 1, 2>`
   in mmvq.cu (whole 128-weight block per lane, PT layout, `ptq1_0_pt_block_dot`). Loop: 257 instructions per block
   per lane, 13 loads (9 activation dwordx4, weights dwordx4 + dwordx2 + 2 ushort); ~162 are the base-3 decode
   (per 4 weights and digit: 2 v_mul_u32_u24, 2 v_and, 1 v_perm). Counters: MemUnitBusy 61-69%, VALUBusy 54-63% on the
   big shapes (output head 85 / 73): latency bound at 2 waves per EU (80-115 VGPRs). Gain estimate <= 10%.
   Plan: cut decode VALU (e.g. derive digits t and t+2 from one multiply by 9, or v_pk_mul_lo_u16), then retry 3 waves per
   EU; measure loop size with `kasm.sh` + `kloop.py` and time with `tgprof.py`. Risks: register pressure flips the
   compiler into serialized loads (watch `s_waitcnt vmcnt` count in the loop). Dead ends: split block over 2 lanes
   (memory-instruction bound, MemUnitBusy 82-90%), LDS digit table, R = 2 rows per lane (14.79 vs 13.41 ms).
2. **Hadamard transform + quantize** - 257 launches per token: PTQ1_0 1.77 ms (9.5%), PQ2_0 1.40 ms (7.9%); 5.5-6.9 us
   each against a ~4 us floor for a small kernel. Gain: ~0.45 ms (2.4%) by folding RMS_NORM + MUL(weight) into
   `fwht_quantize_q8_1` (quantize.cu) for the inputs whose only consumer is the transform (FFN input of every layer, the
   attention-layer attn_norm: 80 of the 129 rms_norm launches, 7 us each); each of the 5 blocks per row recomputes the
   row sum of squares (20 KB read). Match in `ggml_cuda_try_fwht_q8` (ggml-cuda.cu). GDN layers' attn_norm also feeds the
   BF16 alpha / beta mat-vecs, so it stays. Also possible: fold SWIGLU into the ffn_down transform input where gate+up
   are not fused. Risks: the rms_norm reduction order (not bit-identical unless it copies rms_norm_f32<1024>'s order).
3. **ffn gate + up + SWIGLU fusion in the other 24 layers** - 0.24 ms per token (1.3%). Plan: in `ggml_cuda_try_fwht_q8`
   put the q8 rows in a pool block (the `out_aliases_in` path) when a later node's output aliases the transform output,
   and let the GLU fusion's memory check ignore src1 when its q8 copy is elsewhere. Risk: coupling two fusions.
4. **MTP verify mat-vec** - 3-column PQ2_0 kernel 16.8 of 24.8 ms per 3-token pass (68%), 42 us vs 27 us per call for
   one column; ~48% VALU utilization (210 loop instructions for 3 rows x 3 columns x 32 weights). Gain 5-10% of the
   step. Plan: share the activation chunk of a step through LDS across waves (our ysh kernel idea, 4 waves per block),
   or 2 rows x 2 columns per lane split; sweep with `ncprof.py <gguf> 3` (temporary env knob for the shape table).
5. **MTP draft passes** - the MTP head runs Q6_K / Q8_0 on the generic kernel (VEGA20 shape) plus the PQ2_0 output head
   (1 column, ~470 us): ~1.5-2 ms per draft, 2 per step at draft 2 (~10% of a step). Plan: our fork's half-wave kernel
   for Q6_K / Q8_0, rms_norm q8_1 (19d64c7) for the head's norms. Estimate 3-5% of MTP t/s.
6. **Decode flash attention at depth** - after `c82cd0052` 232 us per layer at a 33K context (16 layers = 3.7 ms per
   token, ~17% of a token at d32768): 136 MB of K/V per layer read once = ~590 GB/s. Gain ~1 ms per token at d32K
   (~5%) by tuning the (256, 256, ncols 8) tile config row (nthreads, nbatch_fa, nbatch_K in fattn-tile.cuh, AMD table);
   q8_0 KV halves the bytes (also the context win). Measure with `test-export-graph-ops -c 33280` + `test-backend-ops perf
   --test-file ... -o FLASH_ATTN_EXT` (fine for kernels this long).
7. **Prompt flash attention at depth** - 42.1 ms per layer per 512 tokens at a 33K context, 36% of a pp512 pass at d32768
   (52% at d65536), ~10 TFLOPS of 26.9 fp16 peak. Plan: sweep the (256, 256, ncols 32) row; try ncols2 = 8 padded with
   ncols1 = 4 (K/V tile shared by 6 heads instead of 2, 25% wasted columns). Estimate 10-20% of FA, 4-7% of pp @ d32K.
8. **Prompt MMQ** - ~1030 ms per pp512 pass (84%), ~24 TOPS on all large shapes (45% of dp4a peak), y from VRAM via SGPRs
   at J = 64. Further gains need an inner-loop restructure (our fork: ~41% issue ceiling, LDS-bound x reads).
9. **BF16 alpha / beta GEMMs in prompts** - 96 rocBLAS calls, 34 ms per pp512 pass (2.7%), 0.7 TFLOPS. Plan: one GEMM
   for the pair (n = 96 rows) or an MMF / tile kernel for 5120 x 48 x 512. Estimate ~2% pp.
10. **Small kernels per token** - rms_norm 0.9 ms (129 x 7 us), GDN 0.9 ms (48 x 19 us), alpha / beta pair 0.56 ms,
    host gap ~0.8 ms (tg128 53.3 t/s = 18.8 ms vs 17.8 ms GPU). Each <= 2-3%.

### Lessons not written down elsewhere

- DPP on gfx906 (verified with a probe kernel): `row_shl:n` (0x100+n) gives lane i the value of lane i+n, `row_shr:n`
  (0x110+n) lane i-n, within 16-lane rows; quad_perm 0xB1 = xor 1, 0x4E = xor 2; row_half_mirror 0x141, row_mirror 0x140;
  `ds_swizzle` 0x401F = xor 16 within 32 lanes. A DPP move inside a divergent branch (e.g. `cond ? dpp_a : dpp_b`, which
  C evaluates lazily) reads 0 from the disabled lanes: compute every DPP value unconditionally, then select.
- `test-backend-ops perf` adds ~140 us of host gaps per run on this fork: useless for kernels under ~1 ms, fine for
  long ones. Use rocprofv3 traces (`tgprof.py`, `ncprof.py`, `ppprof.py`) for kernel times.
- `llama-bench -p N` resets the recurrent state each repetition (`scale_f32`, ~0.8 ms per pass): not in real decoding.
- A host+device constexpr helper must not depend on `__gfx906__`: the host pass does not define it (the MMQ tile size
  helper guarded that way gave the host a too small shared-memory size and the kernel overflowed it).
- HIP's `__byte_perm` is a generic C emulation; use `__builtin_amdgcn_perm(y, x, sel)` with the CUDA nibble selector
  widened to bytes. `__vsub4(q, 0x01010101)` on digits 0..2 = `(q + 0x7F7F7F7F) ^ 0x80808080` (no carry).
- Generic index math added to a shared kernel template can cost a lot on unrelated instances (padded GQA: +14% on the
  prompt FA kernel); gate such changes at compile time.
- With MTP (`n_rs_seq > 0`) every pass writes K conv-state snapshot copies: graph fusions that expect one copy never
  match while serving. Check fusions with the MTP server, not only llama-bench.
- `pgrep -f pattern` inside a wait loop matches the loop's own command line: use pid files or `pgrep -x`.
- A llama-server under `rocprofv3` can hang after SIGTERM once the trace is written: `kill -9` it (it holds the GPU lock).
- Bonsai on the raw CSV prompt of mtp-test emits EOS first: the Bonsai mtp-test sends `ignore_eos`.
- `test-export-graph-ops -c N` exports the FA ops at context N: quick FA timing at depth without a prefill.

## Round 3: q8_0 K/V attention (2026-10-26, main session)

Done in the main fork first (gfx906-qwen38 c47e1c3, d8c1986, see its README "Long-context attention"), then ported:

- `6d5bebd31`: decode with a quantized K/V cache used the vec kernel (one block per Q head: each K/V head read 6
  times). On gfx906 with GQA it now uses the tile kernel. PQ2_0 q8_0 K/V: tg128 48.89 -> 52.80, @ d32768 28.11 -> 38.84.
- `310544893`: the tile kernel reads q8_0 K/V directly (no f16 copy of the cache, none allocated).
  PQ2_0 q8_0 K/V: tg128 @ d32768 38.84 -> 45.62, pp512 @ d32768 264.0 -> 252.3 (prompt FA ~15% slower than f16 tiles),
  tg128 52.0 (noise vs 52.8). `GGML_CUDA_NO_FA_Q8_0_TILE=1` restores the copy path.

Max context with q8_0 K/V (same method as above): PQ2_0 233,472 -> 253,952 (262,144 does not fit); Qwen3.8 Q3_K_XL on
the main fork 98,304 -> 106,496. PTQ1_0 was already at the full 262,144. Not re-measured: PQ2_0 + MTP.

KLD, direct vs copy path (PQ2_0, q8_0 K/V): ubatch 1 at the noise floor, ubatch 512 0.000124 (fp16 summation).

## Evaluation (2026-10-29)

Full results, method and fork correctness checks: `docs/gfx906/eval-results.md` on branch `gfx906-qwen38` of the main
fork. Bonsai PQ2_0 on this branch, medium thinking effort, against Qwen3.8-27B-UD-Q3_K_XL on its own gfx906 branch:

| benchmark | Bonsai PQ2_0 | Qwen Q3_K_XL |
|---|---:|---:|
| HumanEval+ / MBPP+ (plus) | 93.3 / 79.4 | 95.1 / 81.0 |
| LiveCodeBench (newest 100) | 66.0 | 79.0 |
| IFEval (prompt strict) | 88.1 | 89.3 |
| MMLU-Pro / MATH-500 / AIME 2025 | 78.3 / 96.5 / 70.0 | 82.3 / 96.0 / 80.0 |
| LongBench v2 | 73.1 | 65.6 |

- At xhigh effort: AIME 90.0 (Qwen 96.7), LiveCodeBench with a 32k budget 71.0 (Qwen 80.0).
- Long-context retrieval up to 250k: every planted fact found at every length; the multi-value and aggregation tasks
  fall off past 100k.
- Correctness: KLD vs stock 0.00013-0.00014 (PQ2_0), PTQ1_0 prompt path matches the CPU reference (stock 0.0033);
  PTQ1_0 gives the same logits as PQ2_0 (same ternary weights). MTP lossless up to rounding: draft 2 = draft 3, every
  split against no-MTP greedy text at a near-tie (top-2 probabilities within 0.02).
- GPU time for the whole Bonsai run: ~30 h (Qwen ~38 h).

Launch commands for the Bonsai configurations (longest context, MTP, evaluation) and what each option does:
`docs/gfx906/server-configs.md` on branch `gfx906-qwen38` of the main fork.
