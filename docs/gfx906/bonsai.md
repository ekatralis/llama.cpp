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
so small but a stock-fork issue worth a look (not investigated yet).

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

## Step 6: results so far (llama-bench, `-fa 1`, f16 KV, 3 reps)

| model | build | pp512 | tg128 | pp512 @ d8192 | tg128 @ d8192 |
|---|---|---:|---:|---:|---:|
| PTQ1_0 | stock | 193.19 | 22.74 | 181.43 | 21.25 |
| PTQ1_0 | gfx906-bonsai `d5dcc70a2` | 192.68 | 48.43 | 181.41 | 45.11 |
| PQ2_0 | stock | 141.01 | 33.75 | 136.24 | 31.99 |
| PQ2_0 | gfx906-bonsai `d5dcc70a2` | 142.41 | 50.18 | 136.38 | 45.91 |

## Not ported (yet) and why

- IQ grid / sign work (0b56028, 2180fea), IQ split paths: the Bonsai models have no IQ tensors.
- MMQ config commits (91861c5, 8e1d244, 3c1147a, bafd5d3, fc5e829, 6e5f055, 34b532d, b99ad83): they build on upstream's
  `mmq-config-gcn.cuh` (#27841), which this fork does not have (GCN uses the RDNA2 table here). Pending the prompt
  profile.
- Tiled transposed concat (19194a1): the fork already has an equivalent (`concat_dim0_transpose_u32`, #309).
- ysh / multi-column MMVQ (697ee81, e7d324f, e64e5f4): for the MTP verify; the Bonsai GGUFs have no MTP layer.
