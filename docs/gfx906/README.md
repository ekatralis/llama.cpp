# llama.cpp gfx906 fork - work log

Goal: get the most performance out of an AMD Radeon VII (gfx906) running
`Qwen3.8-27B-UD-Q3_K_XL.gguf`, then judge which changes could go upstream.

- Fork branch: `gfx906` (from upstream `master` @ `f5027fb`, 2026-10-23): general gfx906 work
- Branch `gfx906-qwen38` (from `gfx906` @ `99d3030`, 2026-10-24): work aimed only at this model and
  config (MTP speculative decoding, small-batch kernels); see "MTP plan" below
- Git identity (repo-local only): Evangelos Katralis <vkatralis@gmail.com>

## Evaluation (2026-10-29)

Quality evaluation of `Qwen3.8-27B-UD-Q3_K_XL` (this branch) against Ternary Bonsai PQ2_0 (branch `gfx906-bonsai`),
plus fork-vs-stock correctness checks: `eval-results.md` (results), `eval-plan.md` (plan and settings). In short:
Qwen leads on LiveCodeBench (79% vs 66%), AIME and MMLU-Pro; the two are within ~2 points on HumanEval+, MBPP+, IFEval
and MATH-500. The fork's logits stay within a few times stock's own batch-size noise, and MTP is lossless up to
rounding (draft 2 = draft 3; every split against no-MTP greedy text is at a near-tie).

## Hardware / software

| Item | Value |
|---|---|
| GPU | AMD Radeon VII, gfx906 (`sramecc+:xnack-`), 60 CUs, 1801 MHz max, 16 GB HBM2 (~1 TB/s peak) |
| CPU | Threadripper 1950X, 16C/32T, 31 GB RAM |
| OS | Ubuntu 24.04.5, kernel 6.8.0-138-generic |
| ROCm | 7.14.0 (`/opt/rocm`), AMD clang 23.0.0git |

## Packages installed

All installed with `apt-get install` as root on 2026-10-23.

| Package | Why |
|---|---|
| `ninja-build` | faster incremental builds |
| `ccache` | cache compiled objects across rebuilds |
| `linux-tools-common`, `linux-tools-generic` | `perf` for CPU-side profiling (built for 6.8.0-147; the `perf` wrapper refuses the running 6.8.0-138 kernel, call `/usr/lib/linux-tools/6.8.0-147-generic/perf`; `perf record` is blocked by perf_event_paranoid = 4) |
| `htop`, `wget` | monitoring, downloads |
| `python3-numpy`, `python3-yaml` | needed by `gguf-py` to inspect model files |
| `jq` | JSON parsing for the Claude Code status line script (not used by the fork) |
| `gdb` (2026-10-24) | stack sampling of llama-server (perf is blocked in this container) |
| `python3.12-venv`, `python3-pip` (2026-10-26) | Python venv `/opt/eval/venv` for the model evaluation (`docs/gfx906/eval-plan.md`): evalplus, math-verify, huggingface_hub, pyarrow, langdetect, nltk, absl-py, immutabledict, numpy, tqdm via pip, inside the venv only |

Models downloaded for the evaluation (2026-10-26): `/opt/llms/ref/Qwen3.8-27B-UD-Q6_K.gguf` (unsloth, 22.0 GB,
sha256 checked against Hugging Face, `SHA256SUMS` next to it).

Repo-local git identity set with `git config user.name/user.email` (not global).

Commit policy (from `AGENTS.md`): commits made by the assistant use an
`Assisted-by: Claude` trailer, never `Co-authored-by`; no automated push or PR.
The owner allowed committing to the `gfx906` branch as work goes on. Commit messages are
assistant-written; upstream does not accept AI-written messages or PR text, so rewrite them
before sending anything upstream.

## Scripts

- `/opt/build-llama.sh`: original upstream build (dir `build/`, make).
- `/opt/build-gfx906.sh`: fork build. Ninja + ccache, tests ON, examples OFF,
  dir `build-gfx906/`. Extra cmake flags pass through: `/opt/build-gfx906.sh -DGGML_X=ON`.

Helper tools in `/opt/gfx906-tools/`:

| File | Use |
|---|---|
| `kstats.py <rocprofv3 .db> [n] [len]` | per-kernel GPU time summary from a `rocprofv3 --kernel-trace` database |
| `ops/qwen38.txt` | all unique ops of the model, from `test-export-graph-ops -m <model> -c 4096 -ub 512 -fa on` |
| `ops/qwen38_tg_mm.txt`, `ops/qwen38_pp_mm.txt` | only the MUL_MAT ops with batch 1 (generation) / 512 (prompt) |
| `mmvq-bench.sh [type regex]` | `test-backend-ops perf` on the batch-1 matmuls, prints `type KxN us GB/s` |
| `sweep-mmvq.sh NW RPB [regex]` | sets the temporary mmvq knobs, rebuilds `test-backend-ops`, runs `mmvq-bench.sh` |
| `kres.sh <src> [regex]` + `kres_parse.py` | VGPR / spill / occupancy per kernel via `-Rpass-analysis=kernel-resource-usage` |
| `kasm.sh <src> <out.s>` + `kloop.py <out.s> <mangled name>` | dump gfx906 assembly, count instruction mix of a kernel's inner loops |
| `mmq-bench.sh [type]` | `test-backend-ops perf` on the batch-512 matmuls (MMQ), prints `type KxN us` |
| `sweep-mmq.sh <CFG_TYPE> <test type> NTHREADS OCC I` | edit the GCN MMQ config at J = 64 for one type, rebuild, run `mmq-bench.sh` |
| `bw/bw.hip` | peak HBM read bandwidth test (`hipcc -O3 --offload-arch=gfx906`) |
| `gpu.sh <cmd...>` | runs a command holding the exclusive GPU lock (`flock`); every GPU job goes through it when two agents share the card |
| `mk-ncols-ops.py <in> <out> n...` | MUL_MAT op file with n columns from the batch-1 ops (env `TYPES`, `SHAPES` filter) |
| `mmvq-ncols-bench.sh [regex] [ns]` | matmul time per column count, e.g. `n1= n2= n3= n4=` per shape (env `TYPES`, `SHAPES`, `BIN` = build bin dir) |
| `mtp-test.sh <label> [server args]` | llama-server t/s and draft acceptance on two greedy prompts |
| `srvprof.py <rocpd db>` | busy / idle / gaps / kernel mix of a profiled llama-server run (raw rocpd tables) |
| `mmstats.py <rocpd db> <n tokens>` | matvec time per (type, columns, kernel variant, launch blocks) from a generation profile |
| `hw-tune.py <label> [ENV=VAL ...]` | profiles a 32-token generation with the given env, prints matvec us/call per (type, rows) |
| `proto/` | standalone single-column MMVQ bench (`build.sh`, `mmvq-proto <type> [filter]`, ~7 s builds), see below |
| `proto/dump-tensors.py <gguf> <dir>` | one raw tensor per (type, shape) of the model into `proto/data/` (909 MB) for the bench |
| `proto/pmc.sh <type> <filter>` | VALUBusy / MemUnitBusy / LDS bank conflicts / TA busy of bench variants (rocprofv3 --pmc) |

Profiling a generation run:
`rocprofv3 --kernel-trace -d DIR -o tg -- llama-bench -m MODEL -fa 1 -p 0 -n 64 -r 1`,
then `kstats.py DIR/tg_results.db`.

## Model

`Qwen3.8-27B-UD-Q3_K_XL.gguf`, arch `qwen35`: 65 layers, a hybrid of gated
delta-net (SSM) layers and full attention every 4th layer; 24 Q heads / 4 KV heads;
context 262144. Despite the name, most weight bytes are i-quants:

| Type | Tensors | GiB |
|---|---:|---:|
| IQ4_XS | 156 | 4.63 |
| IQ3_S | 111 | 3.29 |
| Q5_K | 26 | 1.09 (incl. `output.weight`) |
| IQ3_XXS | 34 | 0.84 |
| Q3_K | 12 | 0.81 (incl. `token_embd`, only read via get_rows) |
| Q4_K | 36 | 0.60 |
| IQ2_S | 15 | 0.38 |
| Q6_K | 7 | 0.32 |
| other (IQ2_XS, Q2_K, IQ2_XXS, IQ4_NL, Q8_0, F32) | | ~0.28 |

So the **IQ4_XS and IQ3_S mat-vec / MMQ kernels matter most**.

## Baseline (upstream f5027fb, `/opt/initial_bench.sh`)

| test | fa=0 | fa=1 |
|---|---:|---:|
| pp512 | 240.6 | 241.0 |
| tg128 | 22.04 | 22.16 |
| pp512 @ d8192 | 215.2 | 223.4 |
| tg128 @ d8192 | 18.36 | 21.45 |

About 11.7 GiB of weights are read per generated token (everything except
`token_embd`), so 22 t/s ≈ 275 GB/s, roughly 27% of HBM2 peak.

## Upstream / related work (checked 2026-10-23)

Local checkout is shallow (1 commit) and `gh` is not installed, so this was read from github.com.

- No upstream PR moves IQ grid tables (`iq3s_grid`, `iq2s_grid`, ...) into LDS, and no PR
  tunes MMVQ nwarps / rows per block for GCN at batch 1.
- #27962 (merged): SWAR `__vsub4` / `__vcmpne4` for IQ2/IQ3 on HIP. Already in our base.
- #30517 (merged Oct 17): faster MMVQ for low-bit types; adds GCN `calc_nwarps` special cases
  for Q3_K / Q6_K at batch 2-4. GCN MMVQ tables were tuned on CDNA (MI100). In our base.
- #27841 (merged): GCN MMQ config (`mmq-config-gcn.cuh`). Drafts #30021 / #30022 tune GCN MMQ
  and stream-k. Draft #30344: fattn tile GCN config. Draft #30348: CI spill check.
- #29910 (closed, not merged): Q2_K MMQ spill fix (gentler unroll, drop temp `y_df` loop).
  Spills depend on ROCm version: ROCm 7.14 baseline 128 VGPR / 146 spills -> 68 / 0 with the PR;
  ROCm 7.2.x baseline only 11 spills. Maintainers on older ROCm saw no gain, so it was closed.
  We use ROCm 7.14, where the spill is real (see log).
- Community fork `sixvolts/llamacpp-gfx906-furnace` (branch `gfx906-perf`): weight repacking
  for K-quants (claims ~89% vs ~58% of HBM BW), DPP warp reductions, LDS gated delta net,
  MMQ LDS layout, "IQ3_S repack" (details not checked yet).

- `milpster/gfx906-llama-cpp` (read 2026-10-23, shallow clone, not built): production fork
  for 2x Radeon VII + RTX 3080 (Vulkan), ROCm 6.1, Q6_K models, long context + speculative
  decoding. No IQ-grid or MMVQ launch-shape work. Their own A/B: fork is 28% slower than
  mainline for deep TG. Useful as a list of measured negatives on gfx906:
  - MMQ Y-tile L2 prefetch (ported from iacopPBK): -7.5% PP.
  - 5 MMQ inner-loop restructures: no gain ("~41% issue ceiling is structural").
  - Chunked (WY) gated delta net prefill: 0.5x vs recurrent (no matrix units).
  - fattn tile scheduling / geometry / XOR swizzle probes: -0.5 to -5.8%.
  - Pointers worth a look later: iacopPBK/llama.cpp-gfx906 (`ggml-cuda/gfx906/`: DPP warp
    reductions, q8 activation cache, fused norm), mixa3607 `amd-memory-tweak` (HBM2 timing
    tuning, needs root, stability risk).
  - Claims ROCm 6.1 is "the measured optimum for gfx906", with no numbers given.

## MTP plan (branch `gfx906-qwen38`, 2026-10-24)

The model has one MTP (nextn) layer; `llama-server --spec-type draft-mtp --spec-draft-n-max 3`
already gives 34-39 t/s vs 28 t/s without (greedy, two prompts, 69-83% draft acceptance).
Each step is one verify pass over k+1 tokens plus k draft passes.

Forward pass time vs batch size (`llama-bench -p N -n 0 -r 40`, median):

| tokens | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| ms | 33.1 | 48.3 | 69.3 | 67.5 | 67.4 |

Weights are read once per pass, so this should be nearly flat. Matmul cost per call
(`/opt/gfx906-tools/mmvq-ncols-bench.sh`, us):

| type / shape | n=1 | n=2 | n=3 | n=4 | n=4 / n=1 |
|---|---:|---:|---:|---:|---:|
| IQ4_XS 5120x17408 | 76 | 134 | 186 | 195 | 2.6x |
| IQ3_S 5120x17408 | 94 | 132 | 258 | 226 | 2.4x |
| IQ3_XXS 5120x17408 | 94 | 138 | 206 | 204 | 2.2x |
| Q4_K 5120x17408 | 82 | 120 | 155 | 215 | 2.6x |
| Q5_K output 5120x248320 | 1298 | 1834 | 2266 | 3073 | 2.4x |

The weight decode is not repeated per column: the compiler shares it (v_perm / grid reads
constant from n=1 to n=4; loop 189 -> 281 instructions for IQ4_XS). What grows are the
activation (y) loads, 3 vector loads per column per iteration. A workgroup covers only 2 rows
but reads the whole y, so a 17408-row matmul with 4 columns pulls ~200 MB of y through the cache
for 47 MB of weights. n=3 is often slower than n=4 (launch config).

Estimate (draft 3; 2.75-3.2 tokens per step measured; ~13.5 ms of draft passes per step):

| scenario | step | t/s |
|---|---:|---:|
| MTP today (verify 67.5 ms) | ~81 ms | 34-39 |
| verify pass at ~1.25x a 1-token pass (~41 ms) | ~55 ms | ~50-58 |
| + cheaper draft passes (output head is 1.3 ms per draft) | ~48 ms | ~57-65 |
| no MTP, single-token matmuls 26.7 -> 20 ms (repacking, uncertain) | ~26 ms/token | ~38 |

So the small-batch (2-5 column) kernels come first. Single-token work still helps the draft
passes. Caveat: acceptance is from two greedy prompts; sampling or harder text lowers it.

Plan: (1) small-batch MMVQ: more rows per workgroup / y reuse, fix n=3; (2) tiled concat for
small token counts; (3) cheaper draft pass.

### MTP progress (2026-10-24)

Cause of the small-batch cost (corrected): not repeated weight decode (the compiler shares it
across columns) but y reads. Counters (IQ3_S/IQ4_XS 5120x17408): DRAM read requests constant
for n = 1/2/4, L2 requests 0.74M -> 1.29M -> 2.52M. Each 2-row block reads all of y.

- Launch sweep of the stock kernel (nwarps 1/2/4, rows 2/4/8): 1 warp best, rows per type;
  still ~2x n=1 at n=4. Not kept.
- New `mul_mat_vec_q_ysh` (gfx906, 2-8 columns, no ids/fusion, K % 512 == 0): 4 waves per
  block share each y chunk in LDS (double buffered, global loads issued before the compute and
  stored after it); each wave owns 2 rows (4 for IQ3_S). L2 requests at n=3 back to ~n=1 level.
  First version spilled (lambdas capturing the prefetch registers by reference): use plain code.
- Narrow dim-0 concat kernel (one thread per element) for the ssm_conv input with 2-31 tokens:
  ~80 us -> 13-32 us.

Forward pass (ms, median of 40): 1/2/3/4/5 tokens 33.1/48.3/69.3/67.5/67.4 ->
33.2/40.4/45.8/54.7/62.5.

llama-server, greedy, 256 tokens, two prompts (t/s):

| config | before | now |
|---|---:|---:|
| no MTP | 28 / 28 | 30 / 30 |
| MTP draft 1 | 35 / 33 | 40 / 38 |
| MTP draft 2 | 33 / 30 | 48 / 45 |
| MTP draft 3 | 39 / 34 | 45 / 41 |
| MTP draft 4 | - | 42 / 38 |

Server step breakdown (draft 2, rocprofv3 kernel + memory copy trace): ~55.8 ms per step =
verify pass (3 tokens) 46.9 + two draft passes 2 x 2.0 (output head ~1.1 each) + MTP catch-up
pass 0.8 + host gaps ~3.3 + copies ~0.5. gdb stack sampling (perf is blocked in this
container: perf_event_paranoid = 4, not writable) shows the server thread waiting on the GPU in
79/80 samples. CPU sampling costs 0.33 ms per token (llama-completion perf print). HIP graphs
off: +1 t/s at most. `--backend-sampling` falls back to CPU: ROCm has no ARGSORT for a 248K
vocab (needed by top-p).

### MTP / generation round 3 (2026-10-24)

- ysh kernel: IQ3_S uses 2 rows per wave below 4 columns (n = 3: 3-7% faster).
  `amdgpu_waves_per_eu` hints (2/3/4) change nothing. More waves per block or rows per wave:
  see sweep in the commit history; 4 waves x 2 rows is best for the other types.
- n = 3 vs n = 1 pass (llama-bench -p N, 30 reps): matmuls +10.6 ms of +11.9 ms. K-quants scale
  worst (Q4_K 1.8x, Q5_K 1.67x): half of the Q4_K dp4a compute per-thread sums of y (the
  m term); the fp16 block sum (ds.y) would avoid them but changes numerics, not done.
- `llama-bench -p N` starts from an empty sequence each repetition, so it zeroes every recurrent
  state each pass (scale_f32 over 786432 floats, ~0.8 ms/pass). Normal decoding does not.
- The recurrent state gather (`k_get_rows_float_vec`, 48 x 12-20 us per pass) is an identity
  copy in the single sequence case. Skipping it needs a non-mutating s_copy, a graph reuse key
  and a fallback after rejected drafts (rollback snapshots); not done.
- MMVQ q8_1 reuse (`d402b46`): matmuls on the same src1 reuse the last quantized copy within a
  graph (176 of 461 quantize launches per token). tg128 30.43 -> 30.94; MTP draft 2: 49 / 46 t/s.
- ROCm env knobs on tg128 (HIP_FORCE_DEV_KERNARG, DEBUG_CLR_GRAPH_PACKET_CAPTURE 0/1,
  GPU_MAX_HW_QUEUES=1, HSA_ENABLE_SDMA=0): no change.
- Full `test-backend-ops`: only TOPK_MOE cases fail, same as stock (which cases fail varies
  per run).

Generation now (31.9 ms GPU per token, 1675 kernels): matmuls 25.1 ms; ~1210 small kernels take
~6.8 ms, nearly all 3.7-5 us each regardless of size (dispatch bound). A delta-net layer has ~20
small kernels (~100 us). Fusion candidates per token: rms_norm + quantize ~0.45 ms, swiglu +
quantize ~0.25 ms, gated norm + silu + quantize ~0.35 ms, l2 norms into the GDN kernel ~0.35 ms,
beta/alpha path (2 tiny matmuls + add + softplus*a + sigmoid) ~0.7 ms.

### Small-kernel fusion round (2026-10-24, branch gfx906-qwen38)

Generation was ~79% matmul; the rest was ~1210 small kernels per token at 3.7-5 us each
(dispatch bound). All changes below keep the logits bit-identical (kld-check.sh 1 and 4 give the
same numbers as before; MTP server outputs compared with an env switch / previous build).

| commit | change | tg128 t/s |
|---|---|---:|
| (before) | | 30.43 |
| d402b46 | reuse the MMVQ q8_1 copy of src1 across matmuls in a graph | 30.94 |
| 6cdfbae | SWIGLU / silu*mul kernels write the q8_1 copy (in-kernel quantize) | 31.15 |
| 18c3d74 | ssm_alpha/ssm_beta: MMVQ epilogue does +bias, softplus*a / sigmoid | 31.39 |
| cfc6d64 | delta net reads the recurrent state cache row via the gather index (GET_ROWS skipped) | 32.06 |
| 052ca53 | conv state shift: concat + state copy-back in one kernel, reading the cache row | 32.54 |

Kernels per token 1675 -> 1295, GPU time per token 31.9 -> 30.2 ms.
After the round: tg128 32.54, tg128 @ d8192 30.81 (was 29.09), pp512 385 (unchanged).
llama-server (two greedy prompts, 256 tokens): no MTP 32.1 / 32.1 t/s; MTP draft 2 50.1 / 46.9;
draft 3 47.0 / 42.6. Full test-backend-ops: only TOPK_MOE (flaky, as stock); model ops 154/161
(same 7 as stock).

Details and negatives:
- In-kernel quantize uses the same butterfly order as quantize_q8_1 (`quantize-lane.cuh`), so the
  q8_1 data is identical. Producers find their consumer by looking ahead 16 graph nodes for a
  MUL_MAT on the same data (possibly a view) and register the buffer in the MMVQ src1 cache.
- rms_norm + quantize was not faster: one 1024-thread block per row, the extra shuffle chain is
  serial (+3.4 us vs a 3.8 us quantize launch). Without the division: 9.4 us, still no win.
  Other rms_norm launch shapes for 5120 columns (256x5, 512x3, 128x16): all slower than 1024x4.
- The gate fusion variant of MMVQ (has_fusion) was 40% slower for the 48-row Q8_0 matmuls (gate
  machinery in the k loop, 954 vs 216 instructions); the post op now runs in the plain variant.
- Deferred GET_ROWS: only for one cache row (one sequence), consumers checked with the graph use
  counts through views; delta net with <= 8 tokens (AR kernel; the MMA path is skipped), or a
  matching conv-state concat. Off with `GGML_CUDA_NO_DEFER_STATE_ROWS=1`. In place is safe:
  each thread loads its state elements before writing them.
- ROCm env knobs, more waves per EU, rope (latency of powf/sinf, 9.9 us for 4 or 24 heads):
  no change / not done.

Remaining per delta-net layer: rms_norm 7, quantize 3.7, concat 5.6, ssm_conv 5.0,
2 x l2 norm 7.7, delta net 13.4, gated norm 4.5 us, plus the matmuls. Not done: l2 norms into the
delta net (bit-exact needs the vec-path reduction order: 4 lanes per partial), ssm_conv into the
concat kernel (needs a memory-overlap check for computing a later node early), gate/beta in one
launch.

### Weight repacking investigation and single-column MMVQ round (2026-10-24, branch gfx906-qwen38)

Started as weight repacking (aligned, wider loads for IQ3_S / IQ4_XS). Result: repacking is not
needed on gfx906; the gain came from how the compiler schedules the existing kernel's loads.
Commit cf35d3b: tg128 32.49 -> 36.30 t/s (+11.7%), tg128 @ d8192 30.93 -> 34.25, pp unchanged,
matvec GPU time per token 25.3 -> 22.2 ms. llama-server: no MTP 32.1 -> 35.7 / 35.6 t/s; MTP draft 2
50.3 / 47.2 (unchanged, the 3-column verify does not use the new kernel).

Prototype (`/opt/gfx906-tools/proto`): one HIP file that includes the llama.cpp headers, runs the
stock kernel shape and variants on real model tensors, checks results against the reference.
Per type a `load()` (raw words) / `dot()` (math) split, loop modes: plain vec_dot calls, explicit
load phase, double buffered, half-wave rows.

Findings:
- `__launch_bounds__(n, 1)` makes the AMDGPU scheduler aim for maximum occupancy; it then issues
  the loads of each row of a step only after the previous row's math (4-5 `s_waitcnt vmcnt(0)`
  per loop instead of 1). `amdgpu_waves_per_eu(4, 4)` + `amdgpu_flat_work_group_size` (without
  launch bounds, whose min-blocks argument overrides it) lets it use 64 VGPRs and keep all loads
  in flight. Fragile: 3-4 more live VGPRs (tried: prefetching the bias / post-mul operands before
  the loop) and it falls back to the serialized schedule (49 VGPRs, iq3_s qkv 41 -> 56 us).
  Check: count `s_waitcnt vmcnt(0)` in the loop body (`kasm.sh` + `kloop.py`).
- Load-only probes (same loads, no math): IQ3_S stock layout ~560-590 GB/s, repacked (per lane
  12 B body = qs + signs, per block 14 B header = scales, qh, d; same size) ~650-680 GB/s.
  IQ4_XS stock layout = repacked, ~750-790 GB/s. Streaming uint4 reads of 13-47 MB: 800-870 GB/s.
- With the loads in flight the full IQ3_S kernel runs at the same speed on both layouts (VALU 66%,
  memory unit 70% busy, rocprofv3 counters). So the limit is the decode (grid lookups in LDS, sign
  masks), and repacking without growing the weights cannot remove it. Sign handling without
  `v_mul_lo_u32`: ~2%. Sign-mask LDS tables (256 x uint2 / uint4): 24% fewer VALU instructions
  but LDS bank conflicts 20% -> 35-41%, slower.
- Half-wave rows: for K = 5120 (20 superblocks) a full wave covers 8 blocks per step, so the last
  of 3 steps is half idle; a half wave covers 4 blocks per step and K = 20, 24, 68 divide evenly.
  +10% on the K = 5120 shapes, neutral on K = 17408.
- The stock vec_dot functions in the half-wave layout are within ~2% of hand-written load/dot
  versions, so the kernel uses them for all types.

Kernel `mul_mat_vec_q_hw` (mmvq.cu): gfx906, 1 column, no ids. Each half wave owns 2 rows
(1 with a fused gate), 4 waves per block share the LDS grid. Epilogue: bias, gate + GLU,
sigmoid / softplus*mul. Selection (measured in the model with `hw-tune.py`): IQ1/IQ2/IQ3 and Q2_K
with >= 2048 rows; IQ4_XS and Q3_K only with a fused gate or > 5120 rows (with 5120 rows and a
cheap decode the stock kernel's 1-wave blocks are 5-10% faster); Q4_K, Q5_K, Q6_K, IQ4_NL, Q8_0
stay stock. Off with `GGML_CUDA_NO_MMVQ_HW=1`. Not bit-exact with stock (summation order):
kld-check.sh 1 gives KLD 0.000384 (was 0.000341), max 0.032, same top 97.84%; ubatch 4 unchanged.

Matvec ms per token (65-token profile), stock kernel -> new kernel:

| type | ms | | type | ms |
|---|---|---|---|---|
| iq3_s | 8.16 -> 6.16 | | iq3_xxs | 2.35 -> 1.85 |
| iq4_xs | 7.72 -> 7.25 | | iq2_s, iq2_xs, iq2_xxs, q2_K, q3_K | 3.10 -> 2.92 |

Fused ffn gate+up per call: IQ3_S 179 -> 125 us, IQ4_XS 145 -> 122 us.

Tried and dropped:
- `amdgpu_waves_per_eu(4, 4)` on the stock 1-column kernel alone: IQ4_XS / Q3_K +3-6%, IQ3_S and
  IQ2_S -4..-10%. On the ysh (2-8 column) kernel: n2 -6%, n3 -12% (it needs more than 64 VGPRs).
- Double-buffered loads (next step's loads before this step's math): slower everywhere (VGPRs).
- Persistent blocks (grid-stride over rows): slower. 8 rows per wave: 126 VGPRs, slower.
- `__builtin_amdgcn_sched_group_barrier` to force the load order: no effect.
- `test-backend-ops perf -o MUL_MAT_VEC_FUSION` with model-size cases reports ~15 us/run for a
  47 MB matrix, unusable; fused paths were measured in the model instead.

Remaining: the multi-column path used by the MTP verify (ysh kernel, 3 columns), the 5120-row
IQ4_XS / K-quant calls (bias fused), the 48-row Q8_0 alpha / beta pair (two 9.5 us launches).

### Half-wave MMVQ for 2-4 columns (2026-10-24, e64e5f4)

For the MTP verify (3 columns with draft 2). Proof of concept in `proto/` first (`NC=3 mmvq-proto <type>`),
then in mmvq.cu: `mul_mat_vec_q_hw` takes 1-4 columns.

- With the stock vec_dot calls per (column, row) the compiler shares the x loads between columns
  (same addresses), but it interleaves the rows' loads with the math again (6-14 `vmcnt(0)` waits
  per loop for IQ3_S / IQ4_XS / IQ2). `mmvq_hw_split<type>`: load all raw x words of the step's
  rows and the y blocks, an empty asm memory barrier, then the math per row for all columns
  (same integer math as vec_dot). Done for IQ3_S, IQ3_XXS, IQ4_XS; other types stay on ysh.
- Launch shape per column count (proto sweep, then model): 2 columns 4 waves x 3 rows per half
  wave at 3 waves/EU; 3 columns the same for the grid types, 2 x 4 at 2 waves/EU for IQ4_XS;
  4 columns 2 x 4 at 2 waves/EU (IQ4_XS 2 x 3, 2 x 4 spills at 128 VGPRs).
- test-backend-ops perf, 2 / 3 / 4 columns, sum over the 18 IQ3_S / IQ3_XXS / IQ4_XS shapes:
  +18 / +21 / +24% (up to +45%; IQ4_XS 17408x5120 at 2 columns -8%, a few 5120-row shapes flat).
- Forward pass of 2 / 3 / 4 tokens (llama-bench -p N): 39.3 / 43.9 / 53.4 -> 35.6 / 39.3 / 45.6 ms.
- llama-server MTP draft 2: 50.3 / 47.2 -> 54.2 / 51.2 t/s; draft 3: 47.0 / 42.5 -> 52.3 / 47.1;
  no MTP unchanged (35.5 / 35.4). tg128 36.1, pp512 386 (unchanged).

3-token pass after this (31.0 ms matvec): still on ysh IQ2_S ~2.0 ms, Q4_K ~1.5, Q3_K ~1.3,
Q8_0 48 rows (alpha / beta, 96 x 12 us) ~1.2, Q5_K ~0.7, IQ2_XS / IQ2_XXS / Q2_K ~1.1.
Split paths for IQ2_S and the K-quants might save ~1.5 ms per pass.

### Multi-column MMVQ for IQ2 / K-quants, delta-net alpha / beta in one launch (2026-10-24, 009bc7d, e192e1e)

Starting point (3-token pass, rocprofv3, `llama-bench -p 3 -n 0 -r 20`): GPU time 37.93 ms, matvec
30.78 ms, ~8.3 ms of it in types still on the ysh kernel: IQ2_S 2.1, Q4_K 1.8, Q3_K 1.3, Q8_0
alpha / beta 1.15 (96 launches x 12 us), Q5_K 0.7, IQ2_XS / Q2_K / IQ2_XXS 1.2.

009bc7d, split load / math paths for IQ2_XXS, IQ2_XS, IQ2_S, Q2_K, Q3_K, Q5_K:
- The split interface now has a per-type y struct and loader (`ys`, `load_y`): the IQ types read one
  full q8_1 block, Q2_K / Q3_K 2 ints from each of 4 blocks, Q5_K 2 ints from each of 2 blocks.
  The Q5_K header (dm + scales, 16 bytes) is one 128 bit load.
- First version: IQ2_XXS +28..41%, Q3_K gate/up +11..31%, Q2_K +17..20%, Q5_K output head +12..16%,
  but IQ2_S / IQ2_XS -15..26% vs ysh. Those two still read their grids (4 / 8 KB) from global
  memory; with the grids in LDS (`grid_size` / `grid()` per split type) they are faster than ysh.
- Launch config sweep (temporary env hook, 10 configs x 7 types x 2-4 columns): most differences
  within 3%. Kept: Q2_K / Q3_K with 2 waves, 4 rows per half wave at 2 columns, 3 at 3 columns
  (Q3_K +12 / +9%, Q2_K 3 columns +19%). Configs with 4 waves at 3-4 waves/EU spill for Q2_K / Q3_K.
- Dropped after the in-model profile: Q4_K (slower than ysh on every model shape, e.g. 5120 rows
  59.5 -> 68 us at 3 columns) and Q5_K below 16384 rows (88 -> 93 us). Lesson: the sweep compared
  configs with each other, not with ysh; check against the old kernel per shape.
- In model, 3 columns: IQ2_S 17408 rows 148 -> 107 us, IQ2_XS 146 -> 105, Q3_K 170 -> 115 and
  89 -> 78, Q2_K 121 -> 86. Matvec 30.78 -> 29.82 ms, GPU time 37.93 -> 37.03 ms per pass.

e192e1e, delta-net alpha / beta pair:
- Graph order per layer: alpha mul_mat, reshape, add(dt), softplus, mul(a), reshape, beta mul_mat,
  reshape, sigmoid. Both are 48 x 5120 Q8_0 on the same activation. The matcher takes the 9 nodes;
  the kernel runs one block per row of both matrices (96 blocks of 4 waves splitting K) and applies
  the epilogues, for 1-4 tokens. Before: 2 launches of 9.5-12 us (6 blocks each, the GPU mostly
  idle) plus, above 1 token, the 4 elementwise ops.
- The first matcher assumed beta first (and the first test built the graph that way, so it passed
  while the model never fused). The profile showed 96 Q8_0 calls still there; the test now builds
  the model's order. Check that a fusion fires in the model profile, not only in the test.
- 48 x 6.4 us per pass, 240 fewer kernels per 3-token pass; GPU time 37.03 -> 35.35 ms.
  tg128 35.9 -> 36.8 t/s (the 1-token path also uses it). Off with `GGML_CUDA_NO_GDN_PAIR=1`.
- Test: `test-backend-ops -o GDN_BETA_ALPHA` (Q8_0 fused 1-4 tokens, 5 tokens and Q4_0 unfused).

Results (both commits):

| | before (f21cdbd) | after (e192e1e) |
|---|---:|---:|
| 2 / 3 / 4-token pass (llama-bench -p N, -r 10) | 35.6 / 39.3 / 45.6 ms | 34.4 / 37.4 / 43.5 ms |
| tg128 / tg128 @ d8192 | 36.1 / 34.2 | 36.9 / 34.7 |
| llama-server, no MTP | 35.5 / 35.4 t/s | 36.2 / 36.3 t/s |
| llama-server, MTP draft 2 | 54.2 / 51.2 t/s | 58.8 / 55.2 t/s |
| llama-server, MTP draft 3 | 52.3 / 47.1 t/s | 56.6 / 51.4 t/s |

pp512 385 / 343 @ d8192 (unchanged). llama-bench at 2-4 tokens varies by +-10 t/s; the rocprofv3 GPU
time per pass is the steadier measure. Tests: MUL_MAT 2445/2445, MUL_MAT_VEC_FUSION 1369/1369,
`ops/qwen38_ncols_mm.txt` 129/129, GDN_BETA_ALPHA 12/12, full suite only TOPK_MOE (78, flaky as stock).

Remaining in the 3-token pass: Q4_K ~1.65 ms and Q5_K below 16384 rows ~0.7 ms on ysh, Q5_K output
head 1.15 ms (1 column there, ~770 GB/s, near the limit). Small fusions not done yet: the q / k L2
norms into the delta-net kernel, ssm_conv into the conv-state concat, the gated norm.

### Small-kernel fusion round 2 (2026-10-25, edb1def, 19d64c7, e919904, a565517)

Generation profile after e192e1e: GPU 26.2 ms per token, matvec ~22 ms, the rest ~4.5 ms in ~1250
kernels per token, most of them 4-5 us (near the fixed cost of a launch on this GPU). So the lever
is fewer kernels. Per-token profiles below: `rocprofv3 --kernel-trace` of `llama-bench -p 0 -n 64`.

- edb1def / e919904: ssm_conv + silu and the q / k L2 norms (rms_norm + scale over 128-wide heads)
  in one kernel, one block per head and token, 1-8 tokens. Sums in the order of ssm_conv_f32 and
  rms_norm_f32<32, 1, do_vec> (float4 per thread, 32-lane reduction). 4.7 us instead of
  4.9 + 2 x 4.0 us per layer. tg128 36.86 -> 37.42. Off with `GGML_CUDA_NO_CONV_L2_FUSION=1`.
  Test: GDN_CONV_L2.
- 19d64c7: rms_norm * mul writes the q8_1 copy for the next MMVQ. 6cdfbae found this not faster
  with one block per row (the quantize serializes); now ncols/1024 blocks per row, each sums the
  whole row in the order of rms_norm_f32<1024, 4> (same scale) and normalizes and quantizes a
  1024-column slice, one value per thread. 4.9 us instead of 6.9 (norm) + 3.8 us (quantize), 129
  times per token. tg128 37.45 -> 38.25 (A/B back to back). Off with `GGML_CUDA_NO_NORM_Q8=1`.
  Test: RMS_NORM_MUL_MAT.
- a565517: the residual add after the 5120-row output projections fused into the 2-4 column hw
  MMVQ (bias column stride as a new fusion arg). `ggml_cuda_mmvq_fuses_bias_multi` mirrors the hw
  dispatch; the dispatch asserts that a multi-column bias without ids never reaches another
  kernel (MUL_MAT_ID passes per-expert biases, the first version of the assert caught those in
  MUL_MAT_VEC_FUSION). 111 of the 127 adds of a 3-token pass (Q4_K / Q5_K stay on ysh).
  Test: MUL_MAT_ADD_RESIDUAL.
- 3-token pass: GPU 33.98 -> 33.43 ms, 1377 -> 1170 kernels. Note: `llama-bench -p 3` starts a new
  sequence each time, which zeroes the recurrent states (96 x 8 us scale_f32, 0.77 ms per pass);
  a MTP verify continues a sequence and does not run them. Pass times in this log include them.
- All four leave the logits unchanged: KLD vs the unfused path at the run-to-run noise floor
  (0.000001, max 0.000059), KLD vs stock at ubatch 1 / 3 / 4 the same as at e192e1e.

Tried and dropped:
- Gated norm (rms_norm * w per head, then silu(z) * norm, with the z mat-vec in between): ran the z
  mat-vec first, then one kernel for norm + gate + q8_1 copy. GPU time -0.22 ms per token, 48 fewer
  kernels, but tg128 unchanged in three alternating A/B runs (38.28 / 38.22 / 38.18 vs 38.22 /
  38.21 / 38.31). Not understood; reverted.
- sincosf instead of sinf + cosf in rope: rope_multi stays at 9.8 us. Its code is ~1900
  instructions with ~20 branches (mrope sections), runs once per layer; likely instruction
  cache misses. Not pursued.

Results (fa=1):

| | e192e1e | a565517 |
|---|---:|---:|
| tg128 / tg128 @ d8192 | 36.86 / 34.73 | 38.40 / 35.60 |
| pp512 / pp512 @ d8192 | 385 / 343 | 385 / 343 |
| llama-server, no MTP | 36.2 / 36.3 t/s | 37.7 / 37.6 t/s |
| llama-server, MTP draft 2 | 58.8 / 55.2 t/s | 60.6 / 56.8 t/s |
| llama-server, MTP draft 3 | 56.6 / 51.4 t/s | 58.2 / 52.7 t/s |

Tests: full suite 18864/18941, only TOPK_MOE (77, flaky as stock); `ops/qwen38_ncols_mm.txt`
129/129. Remaining non-matvec per token (~3.5 ms): delta net 0.64 ms, flash attention 0.41,
gated ops 0.37, rope 0.32 (slow kernel, see above), conv state concat 0.27, gated norm 0.22,
attention q / k norm 0.16, K / V set_rows 0.13. In the 3-token pass also SWIGLU (gate / up
are fused only for 1 column) and the remaining adds.

### Open options for prompt processing (2026-10-25, not started)

Status: pp512 385 t/s, pp512 @ d8192 343 t/s (upstream f5027fb: 241 / 223). 385 t/s x 2 x 27.3 B
weights = ~21 TOPS, ~40% of the dp4a peak (60 CU x 64 lanes x 8 ops x 1.75 GHz = ~54 TOPS). This
counts the whole pass, so the MMQ kernels alone are a little above 40%. Gain estimates below are
guesses, not measurements. Recheck the pp512 profile (rocprofv3 kernel trace) before starting any
of them; the last full profile (2.09 s per pass) is from before changes 6-12.

1. gfx906 MMQ kernel with a new x (weight) tile layout. Expected +10-20% pp, weeks of work.
   - Why: after changes 6 / 7 / 10 (y read from VRAM through SGPRs), the dot loop is limited by
     x reads from LDS. The x tile row stride (2*MMQ_TILE_NE_K + 1 ints, padded against bank
     conflicts) only allows 32-bit `ds_read`. Another gfx906 fork reported a "~41% issue
     ceiling" after five inner-loop restructures, which matches what we see.
   - Ideas: an unpadded, swizzled x layout that allows `ds_read_b64` / `ds_read_b128`; or keep a
     slice of x in VGPRs per thread and stream columns of y (y is already in SGPRs).
   - Risk: mmq.cuh / mmq-load-tiles.cuh are shared by all CUDA / HIP targets, so this is the part
     hardest to upstream. Bit-exact with stock is not guaranteed (summation order).
2. Attention at long context. Expected a few % at 8k, more at 32k+.
   - pp512 drops 11% from depth 0 to 8192. Only 16 of the 64 layers are full attention, the rest
     are gated delta net (whose cost does not grow with depth).
   - First measure pp512 @ d32768 and the flash attention share at depth; then look at the
     FA tile kernel config for GCN (no matrix cores on gfx906).
3. Larger ubatch (`-ub 1024` / `2048`, with `-b` at least as big). A settings check only, no code.
   Might give 5-10% on long prompts (fewer weight passes, more MMQ tiles per CU). Measure with
   `llama-bench -p 2048 -ub 512,1024,2048` and check VRAM use at the target context.

Already tried for prompt (see the 2026-10-23 log): chunked (WY) gated delta net prefill (0.5x of
the recurrent kernel without matrix units), double-buffered y, preloading y for all columns
(spills), IQ3_S sign helper in the loader (noise), 256-thread / occupancy-1 MMQ configs.

Update 2026-10-26: option 2 (attention at depth) was looked at, see the next section. The prompt
FA config sweep found nothing better than the current config.

### Long-context attention (2026-10-26, 595e20a, c47e1c3, d8c1986)

Qwen3.8 27B has 24 Q heads on 4 K/V heads (GQA 6), head size 256, 16 full-attention layers.
Decode attention at depth was much slower than its data size needs, worst with a q8_0 K/V cache,
which is what large contexts in 16 GB need.

- 595e20a (port of the Bonsai branch's c82cd0052): the tile kernel took 2 Q heads per block (the
  largest power of two dividing 6), so each K/V head was read 3 times per step. On gfx906, for 1-4
  tokens and GQA 5-7, one tile of 8 Q heads per K/V head (padding columns skipped).
  Decode FA at depth 32768: 6.90 -> 3.45 ms per token. tg128 @ d8192 35.58 -> 36.46,
  @ d32768 30.29 -> 33.59 t/s (f16 K/V).
- c47e1c3: with quantized K/V and 1-2 tokens the dispatcher picked the vec kernel, one block per
  Q head, so each K/V head was read 6 times (1013 us per layer at depth 32768, ~70 GB/s). On
  gfx906 with GQA it now takes the tile kernel (f16 copy of K/V + one read per K/V head).
  q8_0 K/V: tg128 @ d32768 22.69 -> 28.85 t/s.
- d8c1986: the f16 copy then cost more than the attention (4.43 of 7.9 ms per token at 32768) and
  ~320 MiB of the compute buffer at a 98k context. The tile kernel now dequantizes q8_0 straight
  into its shared-memory tile (`flash_attn_tile_load_tile_q8_0`, `kv_q8_0` template flag):
  - one 34 B block per thread; dword loads plus a byte shift (`v_alignbyte`) since blocks are only
    2-byte aligned; int8 -> fp16 exactly as (0x6400 | (q + 128)) - 1152 with `v_perm`, then one
    fp16 multiply by d, the same values as the f16 conversion.
  - First version spilled 81 VGPRs (scratch 296 B) and was only 1103 -> 556 us. Occupancy 1
    removed the spills (213 VGPRs, 324 us); partial unroll (4) of the KQ and VKQ inner loops at
    occupancy 2 gave 56 VGPRs and 226 us (f16 tiles: 219 us). The same unroll on the HIP 8-head
    f16 instances removed their 5-VGPR spill: 219 -> 210 us, tg128 +0.3.
  - Prompt instances (2 Q heads per tile) too, so no f16 copy is allocated at all; they need the
    partial unroll as well (73 VGPR spill, 2.4x slower without it) and are still ~15% slower than
    f16 tiles (46.1 vs 39.7 ms per layer at depth 32768, 512 tokens).
  - `ggml_cuda_fattn_tile_q8_0_direct()` is used by both the dispatch and the FA buffer size.
    `GGML_CUDA_NO_FA_Q8_0_TILE=1` restores the f16 copy path.

| Qwen3.8 27B Q3_K_XL, t/s | f16 K/V before | f16 K/V now | q8_0 K/V before | q8_0 K/V now |
|---|---:|---:|---:|---:|
| tg128 | 38.35 | 38.62 | 36.52 | 38.05 |
| tg128 @ d8192 | 35.58 | 36.80 | - | 36.41 |
| tg128 @ d32768 | 30.29 | 33.59 | 22.69 | 33.25 |
| pp512 @ d32768 | - | 253.2 | 255.8 | 245.0 |
| max llama-server context (16 GB, 4096 steps) | 57344 | - | 98304 | 106496 |

"before" = aab860d (q8_0: vec kernel for decode, f16 copy for prompt). MTP draft 2 / 3 at short
context unchanged (61.1 / 56.8, 57.8 / 52.4 with f16 K/V; q8_0 K/V 60.6 / 54.1).

Output: decode (ubatch 1 / 3) with direct q8_0 equals the f16 copy path (KLD at the noise floor).
Prompt (ubatch 512) differs from it by KLD 0.0006 (fp16 summation, the two instances are compiled
differently; FLASH_ATTN_EXT errors vs CPU are the same size for both), vs stock with f16 K/V:
copy path 0.000748, direct 0.000807 (both mostly the q8_0 cache itself). f16 K/V decode after
595e20a: KLD vs stock 0.000326 / 0.000322 at ubatch 1 / 3 (was 0.000298 / 0.000301).

Tested and not kept: prompt FA configs for 32 columns (512 threads, occupancy 1, nbatch_fa 64,
nbatch_K 64 / 256): none beat 256 / 2 / 32 / 128 at 512 tokens. 512 threads / occupancy 1 /
64 / 64 made the 3-token verify 15% faster (507 -> 433 us at depth 32768) but prompts 1% slower;
the row is shared, so splitting it needs a dispatch change (~2-3% of MTP t/s at 32k, not done).

Open: the 3-token MTP verify FA costs 2.3x one token at depth (477 vs 210 us at 32768) although it
reads the same K/V once; prompt FA with q8_0 is 15% slower than f16 (a loader that stages raw
q8_0 rows through LDS with 16-byte loads might close it).

### Output reference points (vs stock upstream f5027fb)

Measured with `kld-check.sh` (KL divergence of the logits vs stock; ubatch 512 = prompt kernels,
1 = single-token generation kernels, 4 = multi-token kernels as in the MTP verify).

| commits | prompt (ubatch 512) | generation (ubatch 1) |
|---|---|---|
| f5027fb (upstream base) | identical | identical |
| a25d385 .. 642a6b6 | same as stock (KLD 0, max = stock run-to-run noise) | differs: KLD 0.000341, same top 97.6% |
| cf35d3b .. | same as stock (rechecked at 97d07dd) | differs: KLD 0.000384, same top 97.8% |
| e64e5f4 .. | same as stock | ubatch 1 as cf35d3b; ubatch 3 / 4: KLD 0.000348 / 0.000328 |
| 009bc7d, e192e1e .. | same as stock (rechecked at e192e1e) | ubatch 1 / 3 / 4: KLD 0.000298 / 0.000301 / 0.000299, same top 97.8-98.6% |
| edb1def .. a565517 | same as stock | same as e192e1e (KLD vs the unfused path at the noise floor) |

- Generation output has not been bit-identical to stock since the first fork commit a25d385
  (batch-1 MMVQ launch shape 2 -> 1 warps changes the float summation order). The size is the same
  as stock's own difference between ubatch sizes (stock ubatch 4 vs 1: KLD 0.000312).
- a25d385 .. 642a6b6 (tag `gfx906-qwen38-pre-hw`): every commit after a25d385 kept the logits
  identical to the previous fork build (same kld-check numbers, identical MTP outputs).
- cf35d3b (half-wave single-column MMVQ) changes the generation results again (summation order).
  `GGML_CUDA_NO_MMVQ_HW=1` restores the 642a6b6 results. ubatch 4 results did not change.
- e64e5f4 (half-wave MMVQ for 2-4 columns) changes the multi-token results (ubatch 2-4, MTP
  verify); ubatch 4 KLD vs stock 0.000323 -> 0.000328. Also off with `GGML_CUDA_NO_MMVQ_HW=1`.
- 009bc7d (IQ2 / K-quant multi-column paths) changes ubatch 2-4; e192e1e (alpha / beta pair, split K
  summation) changes ubatch 1-4 (`GGML_CUDA_NO_GDN_PAIR=1` turns it off). Measured together.

## Log

### 2026-10-23
- Installed the packages above, created branch `gfx906`, added `/opt/build-gfx906.sh`.
- Inspected the model's quant mix (table above).
- Power state: `auto`, `--setperflevel high` (did not apply) and `COMPUTE` profile all give
  the same numbers; the GPU boosts by itself under load. Not a bottleneck.
- Generation profile (rocprofv3, 65 tokens): ~45 ms GPU time per token = 22 t/s, so launch
  gaps are not the issue. ~85% of time is `mul_mat_vec_q` (MMVQ). By type:
  IQ3_S ~35%, IQ4_XS ~21%, IQ3_XXS ~9%, Q5_K ~5%, IQ2_S ~5%, Q4_K ~3%.
- Per-shape MMVQ bandwidth (batch 1): IQ4_XS / Q4_K / Q5_K 400-530 GB/s;
  IQ3_S / IQ3_XXS ~180-200 GB/s; IQ2_* 150-180 GB/s; Q3_K ~300 GB/s.
- Why IQ2/IQ3 are slow: their grids (`iq3s_grid` etc.) are `static const __device__`
  tables in global memory. Each thread does 8 scattered 4/8-byte loads per vec_dot.
  The vector memory unit serves ~1 cache line per clock per CU for scattered lanes:
  60 CU x 1.8 GHz = ~108 G lookups/s x 4 weights x 3.44 bit = ~186 GB/s, matching the
  measured ~190 GB/s. IQ4_XS has no such problem: its 16-entry table is held in
  registers and indexed with `v_perm_b32`.
- MMVQ launch shape sweep (GCN path, batch 1; temporary knobs `GFX906_MMVQ_NW` /
  `GFX906_MMVQ_RPB` in `mmvq.cu`). Upstream GCN default is 2 warps, 1 row per block.
  More warps (4, 8) is slower. 1 warp x 2 rows is best for every type seen:
  IQ4_XS -15..22%, Q4_K/Q5_K/Q3_K ~-17%, output Q5_K 1584 -> 1300 us, IQ3_S -7%.
- Prompt side: Q2_K at batch 512 takes ~33 ms vs ~4-7 ms for other types of the same
  shape. `kres.sh` shows Q2_K `mul_mat_q` with `mmq_x=64` spills 145 VGPRs (580 B
  scratch / lane); other types do not spill (Q4_K mmq_x=64: 10-26).
- Change 1 (`mmvq.cu`): new MMVQ table id `MMVQ_PARAMETERS_VEGA20` (gfx906 only; GCN4,
  gfx900 and CDNA keep the old GCN values). Batch 1: 1 warp, 2 rows per block.
  Second sweep (1x1, 1x4, 2x4, 1x8) confirmed 1x2 as the best overall.
  Result: tg128 22.17 -> 23.94 t/s, tg128 @ d8192 21.45 -> 23.10. pp unchanged.
- Change 2 (`vecdotq.cuh`, `mmvq.cu`): IQ3_S on gfx906.
  - `iq3s_grid` (2 KB) is copied to LDS once per MMVQ block; the IQ3_S vec_dot takes the
    grid as a pointer (`vec_dot_iq3_s_q8_1_grid`), the old function wraps it with the global table.
  - The sign application (`__vcmpne4` + `__vsub4`, both emulated on HIP) was the next
    bottleneck: the IQ3_S inner loop had ~430 instructions per 64 weights, mostly and/xor/shift.
    New helper `iq_grid_apply_signs4`: spread 4 sign bits to byte LSBs with one
    `v_mul_u32_u24` (`s * 0x00204081 & 0x01010101`), then `(v ^ mask) + lsb`. Valid because all
    IQ grid bytes are in [1, 127] (checked for iq3s, iq3xxs, iq2s, iq2xs, iq2xxs grids), so the
    negation never carries into the next byte. HIP only; CUDA path unchanged.
  - Effect on IQ3_S 17408x5120 (us): stock 164.6; signs only 162.5; LDS only 117.7;
    both 82.9 (~2x, 431 GB/s). The two changes remove two different bottlenecks.
  - More rows per block with the LDS grid (4, 8) is slower, so the per-block table copy is cheap.
  - Result: tg128 23.94 -> 28.73 t/s, tg128 @ d8192 23.10 -> 27.51. pp unchanged.

- Change 3 (`vecdotq.cuh`, `mmvq.cu`): same treatment for the other IQ grid types.
  - Sign helper moved next to `unpack_ksigns` and used for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS
    (HIP only; the 7-bit ksigns + parity byte maps to the same nibble-per-4-bytes layout).
  - LDS grid generalized: `mmvq_lds_grid_size(type)` (gfx906 only) + `get_vec_dot_q_grid_cuda` +
    `get_iq_grid`. Enabled for IQ2_XXS (2 KB), IQ3_XXS (1 KB), IQ3_S (2 KB). `__align__(16)` for
    the 8-byte IQ2 grid reads.
  - IQ2_S (8 KB grid) and IQ2_XS (4 KB) are faster with global grids + sign helper only: with
    K = 5120 the per-block grid copy outweighs the ~3 KB of weights a block reads.
  - IQ3_XXS 17408x5120: 152.5 -> 81 us. IQ2_S 139.6 -> 120.6 us. IQ2_XS 120.9 -> 105.8 us.
  - Result: tg128 28.73 -> 30.49 t/s, tg128 @ d8192 27.51 -> 29.18.
- Peak HBM read bandwidth measured with a simple uint4 read kernel (`/opt/gfx906-tools/bw/bw.hip`):
  ~865 GB/s. So the floor for 12.6 GB of weights per token is ~14.5 ms (~69 t/s).
  At 30.5 t/s (~32.5 ms/token): IQ4_XS runs at ~75% of peak, IQ3_S ~50%, IQ3_XXS ~45%.
- Tried and rejected for MMVQ (all measured with `mmvq-bench.sh`):
  - `v_perm_b32` instead of `lsb*255` for the sign mask: removes 16 `v_mul_lo_u32` per loop,
    no measurable change. IQ3_S loop is now ~237 instructions per 64 weights (was ~430).
  - VDR 4 for IQ3_S (each thread does two 2-int chunks): 83 -> 114 us. More threads in flight win.
  - Launch shape re-sweep for LDS-grid types (1x2, 2x1, 2x2, 4x1, 4x2, 2x4): 1x2 still best.
  - 2 KB LDS sign-mask table for IQ3_S (256 x uint2, built per block): long rows 83 -> 79 us,
    K = 5120 rows 91 -> 107 us. Net loss for this model.
  - Half warp per row (so K = 20/24/68 superblocks have no idle tail): IQ3_XXS -10..13%, but
    K-quants +7..23% slower (the two rows no longer share y loads). Net loss.
- Prompt processing profile (pp512, ~2.09 s): MMQ IQ4_XS 619 ms, IQ3_S 615 ms, IQ3_XXS 175 ms,
  Q2_K 99 ms (only 3 tensors), IQ2_S 99 ms, gated delta net 91 ms, Q4_K 78 ms, Q3_K 66 ms.
  All MMQ types run at ~14-15 TOPS (~28% of dp4a peak), with or without grid lookups.
- Change 4 (`mmq-config-gcn.cuh`): Q2_K MMQ tile width capped at 40 on GCN (removed the J = 48
  and J = 64 entries). Under ROCm 7.14 these spill up to ~150 VGPRs: the 512-thread config with
  occupancy 2 caps each thread at 128 VGPRs. Tried first and rejected: dropping the `y_df` array
  (no change), limiting k01 unroll (no change), 256 threads / occupancy 1 / I = 128 (all slower).
  Result: Q2_K 5120x12288 x 512: 33.1 ms -> 7.4 ms.

- Change 5 (`mmq-config-gcn.cuh`): MMQ config sweep at J = 64 (nthreads x I x occupancy:
  512/128/2 (old), 512/64/2, 256/64/2, 256/128/2, 256/128/1, 512/128/1).
  IQ4_XS: 512/64/2 best, -2..-6% on 4 of 6 shapes, flat on the rest -> adopted.
  IQ3_S: old config (512/128/2) is best. Occupancy 1 vs 2 makes no difference at 512 threads.
- Sign helper in the IQ3_S MMQ tile loader: ~1% (noise). MMQ is bound by its dot loop, not by
  tile loads. Not kept.
- IQ4_XS MMQ inner loop (J = 64): 549 instructions, only 128 `v_dot4_i32_i8` (23%); 90 LDS reads,
  76 `v_add_u32` (address math), 45 waits, ~50 for scale conversion. The odd x-tile row stride
  (2*MMQ_TILE_NE_K + 1 ints, to avoid bank conflicts) blocks 64/128-bit LDS reads. This matches
  the "~41% issue ceiling" another fork reported. Raising it needs an MMQ restructure for gfx906.

- Change 6 (`mmq-vec-dot.cuh`): 128 bit LDS reads for y in the q8_0 dp4a vec dot (HIP). y is
  16 byte aligned (column stride 144 B). `ds_read2_b32` offsets (max 1020 B) cannot span the
  1152 B per-warp column step, so each column needed address adds; `ds_read_b128` has a 16 bit
  offset. Loop 549 -> 467 instructions, `v_add_u32` 76 -> 18. IQ3_S -12.7%, IQ4_XS -6.7%.
  Tried after: preload y for all 8 columns (spills, 3-4x slower), double-buffer the next column
  (IQ3_S -3%, IQ4_XS +1%; not kept).
- Diagnosis: y is the same for every lane of a warp (one column per warp), so each 128 bit LDS
  read is a 64-lane broadcast that still costs ~8 LDS clocks. Per k01 step a wave spends ~160 LDS
  clocks vs 256 SIMD clocks of dp4a, and 4 SIMDs share one LDS: LDS bound by ~2.5x.
  Check: replacing the y reads with constants (wrong results) made IQ4_XS 31% faster.
- Change 7 (`mmq.cuh`, `mmq-vec-dot.cuh`, `mmq-config-gcn.cuh`): on gfx906, for types whose dp4a
  vec dot is `ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a` (compile-time function pointer compare), skip
  the y copy to LDS and read y directly from VRAM in `ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a_vram`.
  The column index goes through `__builtin_amdgcn_readfirstlane` and y is read through an
  `address_space(4)` pointer, so the compiler emits `s_load_dwordx8` and dp4a takes y from SGPRs.
  The global y tile has the same layout as the LDS copy (the copy is linear).
  - Limited to `J/nwarps <= 8`: Q8_0 uses J = 128 at n = 512 (16 columns per warp = 128 SGPRs
    for y) and was 43% slower with the VRAM path.
  - With y out of LDS, I = 64 beats I = 128 at J = 64: set for Q5_0, IQ2_XXS, IQ2_XS, IQ2_S,
    IQ3_XXS, IQ3_S, IQ4_NL (IQ2_XS/IQ2_S use the q8_0_16 vec dot, not the VRAM path, but also
    gain 6-11% from I = 64). Q8_0 stays at I = 128.
  - vs previous commit at n = 512: IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL, IQ2_XXS, Q5_0 -22..-32%;
    IQ2_S / IQ2_XS -6..-11%; K-quants, Q4_0, IQ1_S, Q8_0 unchanged.
  - Upstream note: the I = 64 config rows are in the shared GCN table; on other GCN chips (no
    VRAM y path) I = 64 is slower. An upstream version must gate these rows to gfx906.
- Test baseline: `test-backend-ops --test-file ops/qwen38.txt` fails the same 7 non-matmul cases
  (SCALE, CPY, GET_ROWS, GATED_DELTA_NET; NaN/inf in both CPU and GPU from random state inputs)
  on stock upstream f5027fb, built in the worktree `/opt/llama.cpp-stock` (kept as an A/B baseline,
  `build/` there has `test-backend-ops` and `llama-bench`).

- Change 8 (`mmq-load-tiles.cuh`): `iq_grid_apply_signs4` in the IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ3_S MMQ tile loaders (HIP). Now that the dot loop is faster: -2..-3% for each IQ type.
- Change 9 (`mmq.cu`): MMQ tile width (J) choice on GCN. GCN has no stream-k, so a 5120x48 Q8_0
  matmul at n = 512 ran as 4 blocks on 60 CUs (512 us). If the usual J gives fewer than 3 tiles
  per CU (dense only), pick the J that minimizes ceil(ntiles / nCU) * (J + 32).
  A first version that always used this cost model made big IQ matmuls much slower (it picked
  J values with untuned configs). Q8_0 5120x48: 543 -> 228 us; 5120x1024 K/V shapes: -20..-30%.
- Change 10 (`mmq.cuh`, `mmq-vec-dot.cuh`, `mmq-config-gcn.cuh`): the VRAM y path generalized with a
  `y_vram` template flag on the q8_0, q8_0_16, q3_K, q4_K, q5_K, q6_K dp4a vec dots (helpers
  `mmq_y_col` / `mmq_y_regs`; without the flag the code is the same as before).
  `ggml_cuda_mmq_get_vec_dot_y_vram` maps a vec dot to its VRAM variant (compile-time pointer
  compare). Configs at J = 64 for Q3_K, Q4_K, Q5_K, Q6_K: 512 threads, I = 64 (8 columns per warp).
  vs previous commit: IQ2_S / IQ2_XS -29..-31%, Q3_K -40%, Q5_K -20%, Q4_K -14%,
  Q6_K -13% (standard 4096x14336 shape). Q2_K (irregular vec dot) not done.

- Change 11 (`gated_delta_net.cu`): load k one token ahead (k is needed first in each token and
  does not depend on the state). First try prefetched k, q, v, beta, g: VGPRs 40 -> 67, occupancy
  6 -> 3, 32-head cases up to 16% slower. k only: 54 VGPRs, occupancy 4. This model (48 heads,
  512 tokens): 1.99 -> 1.29 ms per layer; 4 heads -38..-41%; 32 heads -4..+4%.
- Change 12 (`concat.cu`): `concat_non_cont` for the ssm_conv input ([3,10240] + transposed
  [512,10240]) read src1 with a 40 KB stride: 678 us, 58 GB/s. New tiled kernel (32x32 tiles via
  shared memory) for dim 0 with src1 contiguous in dim 1 and src1->ne[0] >= 32: 131 us. The size
  limit keeps the 1-token case ([3,10240] + [1,10240]) on the old kernel (tiled was 9.5 -> 13.8 us).

### Generation round 2 (2026-10-23): IQ3 MMVQ, no change kept

Per-token profile (tg, 33 ms/token): MMVQ ~26.7 ms. IQ3_S 9.2 ms (~380 GB/s), IQ4_XS 8.4 ms
(~590 GB/s), IQ3_XXS 2.6 ms, Q5_K 1.9 ms, IQ2_S 1.7 ms (~240 GB/s), Q4_K 1.2 ms;
quantize_q8_1 1.7 ms (461 launches), rms_norm 1.3 ms. GPU is busy 32.5 of 33.5 ms per token,
HIP graphs on or off give the same tg (30.4 vs 30.5): launch overhead is not the limit.

Counters (`rocprofv3 --pmc VALUBusy MemUnitBusy MemUnitStalled` on the tg shapes):

| type | MemUnitBusy (TA) | VALUBusy |
|---|---:|---:|
| IQ3_S | 58-71% | 51-66% |
| IQ3_XXS | 52-63% | 55-71% |
| IQ4_XS | 57-67% | 48-66% |
| IQ2_S | 79-91% | 39-47% |
| IQ2_XS | 88-90% | 52-61% |

IQ3_S and IQ4_XS both process ~1e12 weights/s; IQ3 only looks slower in GB/s because it has
fewer bits per weight. Neither unit is saturated for IQ3. IQ2_S/IQ2_XS are fetch bound
(global grid lookups).

Tried, no gain (all reverted):
- v_perm sign mask (the compiler turns `(lsb << 8) - lsb` into a quarter rate v_mul_lo_u32):
  same e2e and same microbench once the GPU state was stable. A first microbench baseline in
  this round was ~10% slow (GPU state), which made this look like a 6-16% win. Repeat baselines.
- (grid, negated grid) pairs in LDS so one v_perm applies the signs: -18 instr/iter, but the
  doubled LDS grid made IQ3_S up to 20% slower.
- Several row pairs per workgroup (grid copy once): 1/2/4 groups, no gain. A "no grid copy"
  timing test looked 2x faster, but it was invalid: with LDS never written the compiler may drop
  the LDS reads and dependent math.
- 4 or 8 rows per workgroup: mixed (IQ3_XXS up to -18%, IQ3_S up to +10%).
- `#pragma unroll 2` on the MMVQ k loop: 0-15% slower.
- LDS grid for IQ2_XS (4 KiB): 17408x5120 -18%, 5120x17408 +7%; type is 0.3 ms/token, dropped.
- LDS grid for IQ2_S (8 KiB): up to 1.9x slower (8 workgroups per CU).

MTP speculative decoding (model has 1 nextn layer, `--spec-type draft-mtp`), llama-server,
greedy, 256 tokens, two prompts: no MTP 28 t/s; draft max 1: 33-35; 2: 30-33; 3: 34-39 t/s
(acceptance 69-83%).

### Output checks (2026-10-23)

`/opt/gfx906-tools/kld-check.sh [ubatch]`: llama-perplexity KL divergence of the fork vs stock
upstream on ~900 KB of repo docs (`/opt/gfx906-tools/kld-text.txt`), c = 512.

| comparison | mean KLD | max KLD | same top token | PPL ratio |
|---|---:|---:|---:|---:|
| fork vs stock, ubatch 512 (MMQ), 8 chunks | 0.000000 | 0.000059 | 100.0% | 1.000000 |
| stock vs stock, ubatch 1 (noise floor) | 0.000000 | 0.000048 | 100.0% | 1.000000 |
| fork vs stock, ubatch 1 (MMVQ), 2 chunks | 0.000341 | 0.003774 | 97.6% | 1.0012 +- 0.0012 |
| stock ubatch 512 vs stock ubatch 1 | 0.000712 | 0.009195 | 97.3% | 1.0007 +- 0.0015 |
| stock ubatch 4 vs stock ubatch 1 | 0.000312 | 0.004705 | 97.3% | - |

- The prompt (MMQ) path is bit-exact vs stock. All IQ sign / LDS-grid changes are bit-exact too
  (bisect: KLD unchanged across those commits).
- The ubatch-1 difference comes only from the MMVQ launch shape (nwarps 2 -> 1; rows per block
  does not matter): float summation order. Isolated, each batch-1 matmul has the same error vs CPU
  with 1 or 2 warps (~2.5e-5 NMSE, from the q8_1 activation quantization). The size of the
  difference is the same as stock's own difference between ubatch sizes, so it is not a bug.
- `test-backend-ops --test-file` with per-test errors: `/opt/gfx906-tools/test-backend-ops-print-err.patch`
  (adds `GFX906_PRINT_ERR=1`), parse with `/opt/gfx906-tools/errs.py`.
- Greedy generation (200 tokens) is identical to stock for ~150 tokens, then a near-tie flips.

### Results so far (fa=1, llama-bench -r 3)

| build | pp512 | tg128 | pp512 @ d8192 | tg128 @ d8192 |
|---|---:|---:|---:|---:|
| upstream f5027fb | 241.0 | 22.16 | 223.4 | 21.45 |
| + VEGA20 MMVQ shape | 241.1 | 23.94 | 224.5 | 23.10 |
| + IQ3_S LDS grid + sign math | 241.2 | 28.73 | 224.5 | 27.51 |
| + IQ2/IQ3_XXS signs, LDS grids | 241.3 | 30.49 | 224.5 | 29.18 |
| + Q2_K MMQ J cap, IQ4_XS MMQ I = 64 | 254.2 | 30.45 | 235.3 | 29.12 |
| + MMQ 128 bit y reads, y from VRAM via SGPRs | 339.3 | 30.46 | 306.2 | 29.15 |
| + IQ sign helper in MMQ loaders, MMQ J rebalance | 353.9 | 30.47 | 317.4 | 29.15 |
| + y from VRAM for K-quant / q8_0_16 vec dots | 372.0 | 30.48 | 331.8 | 29.15 |
| + GDN k prefetch, tiled transposed concat | 386.4 | 30.41 | 343.3 | 29.09 |
| branch gfx906-qwen38 at e192e1e (fusions, half-wave MMVQ, alpha / beta pair) | 385.3 | 36.86 | 342.6 | 34.73 |
| branch gfx906-qwen38 at a565517 (fusion round 2) | 385.2 | 38.40 | 342.9 | 35.60 |

Tests after each change: `test-backend-ops test -b ROCm0 -o MUL_MAT,MUL_MAT_ID` (3376/3376)
and `--test-file ops/qwen38_tg_mm.txt` (43/43).
