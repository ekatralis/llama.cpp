# llama.cpp gfx906 fork - work log

Goal: get the most performance out of an AMD Radeon VII (gfx906) running
`Qwen3.8-27B-UD-Q3_K_XL.gguf`, then judge which changes could go upstream.

- Fork branch: `gfx906` (from upstream `master` @ `f5027fb`, 2026-10-23)
- Git identity (repo-local only): Evangelos Katralis <vkatralis@gmail.com>

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
| `linux-tools-common`, `linux-tools-generic` | `perf` for CPU-side profiling (built for 6.8.0-147, may warn on the running 6.8.0-138 kernel) |
| `htop`, `wget` | monitoring, downloads |
| `python3-numpy`, `python3-yaml` | needed by `gguf-py` to inspect model files |
| `jq` | JSON parsing for the Claude Code status line script (not used by the fork) |

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

### Results so far (fa=1, llama-bench -r 3)

| build | pp512 | tg128 | pp512 @ d8192 | tg128 @ d8192 |
|---|---:|---:|---:|---:|
| upstream f5027fb | 241.0 | 22.16 | 223.4 | 21.45 |
| + VEGA20 MMVQ shape | 241.1 | 23.94 | 224.5 | 23.10 |
| + IQ3_S LDS grid + sign math | 241.2 | 28.73 | 224.5 | 27.51 |
| + IQ2/IQ3_XXS signs, LDS grids | 241.3 | 30.49 | 224.5 | 29.18 |
| + Q2_K MMQ J cap, IQ4_XS MMQ I = 64 | 254.2 | 30.45 | 235.3 | 29.12 |
| + MMQ 128 bit y reads, y from VRAM via SGPRs | 339.3 | 30.46 | 306.2 | 29.15 |

Tests after each change: `test-backend-ops test -b ROCm0 -o MUL_MAT,MUL_MAT_ID` (3376/3376)
and `--test-file ops/qwen38_tg_mm.txt` (43/43).
