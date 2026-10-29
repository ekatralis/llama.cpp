# llama-server configurations used on the Radeon VII (gfx906)

The `llama-server` launch commands used for serving, speed measurements and the evaluation in this work, with what
each option does. Numbers are from `README.md`, `eval-results.md` and the Bonsai fork's `docs/gfx906/bonsai.md`.
The evaluation's own launcher is `/opt/gfx906-tools/eval/serve.sh` (all of the evaluation commands below are what it
runs).

## Binaries

| build | path | built by | used for |
|---|---|---|---|
| this fork, branch `gfx906-qwen38` | `/opt/llama.cpp/build-gfx906/bin/llama-server` | `/opt/build-gfx906.sh` | Qwen3.8, and any other non-Bonsai model |
| upstream stock (`f5027fb`) | `/opt/llama.cpp-stock/build/bin/llama-server` | worktree build | comparisons, Q6_K reference |
| Bonsai fork, branch `gfx906-bonsai` | `/opt/llama.cpp-bonsai/build-gfx906/bin/llama-server` | `BUILD_DIR=build-gfx906 /opt/build-bonsai.sh` | Ternary Bonsai (PQ2_0 / PTQ1_0 need PrismML's types) |
| PrismML stock (`prism`) | `/opt/llama.cpp-bonsai/build-stock/bin/llama-server` | `BUILD_DIR=build-stock /opt/build-bonsai.sh` | Bonsai comparisons |

The stock upstream build cannot load the Bonsai files (PQ2_0 / PTQ1_0 are PrismML types).

## The common command

```bash
/opt/llama.cpp/build-gfx906/bin/llama-server \
    -m /opt/llms/Qwen3.8-27B-UD-Q3_K_XL.gguf \
    -ngl 99 -fit off -fa on -ctk q8_0 -ctv q8_0 \
    -c 98304 -np 4 \
    --host 127.0.0.1 --port 8100 --no-webui \
    --reasoning-format deepseek \
    --reasoning-budget-message $'\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n'
```

| option | what it does | why this value |
|---|---|---|
| `-m FILE` | model GGUF | |
| `-ngl 99` | number of layers offloaded to the GPU; 99 = all | everything runs on the Radeon VII |
| `-fit off` | turns off automatic fitting, which otherwise lowers unset `-c` / `-ngl` until the model fits in VRAM | we choose the context ourselves; with `off` the server fails at start if it does not fit, instead of silently shrinking |
| `-fa on` | flash attention | needed for a quantized V cache; also faster at long context. Our fork's tile kernel reads q8_0 K/V directly |
| `-ctk q8_0 -ctv q8_0` | K and V cache type (default f16) | halves the cache memory per token, so about twice the context fits; quality cost is small (Qwen prompt KLD vs an f16 cache ~0.0008) |
| `-c N` | total context, in tokens, shared by all slots | 98,304 = 4 slots x 24,576 |
| `-np N` | number of parallel slots (requests decoded together) | 4 gives the best total throughput (72 t/s vs 35.5 with 1 slot); each slot gets `c / np` tokens (`kv_unified` is off) |
| `--host 127.0.0.1 --port P` | listen address | local only; the evaluation sandbox has no network |
| `--no-webui` | no browser chat UI | API only |
| `--reasoning-format deepseek` | puts the thinking text in `message.reasoning_content` instead of `content` | scorers read only the answer; thinking length is counted separately |
| `--reasoning-budget-message TEXT` | text inserted before the end-of-thinking tag when the thinking budget runs out | the wording from the Qwen3 report, so the model moves on to an answer |
| `--reasoning-budget N` | server-wide default thinking budget (-1 = none) | not set; the evaluation sends a budget per request (below) |

## Configurations

### Qwen3.8 27B Q3_K_XL (this fork)

| use | command (after the common options `-ngl 99 -fit off -fa on -ctk q8_0 -ctv q8_0`) | measured |
|---|---|---|
| one user, longest context | `-c 106496 -np 1` | largest context that starts and answers (q8_0 K/V); generation ~38 t/s, ~33 t/s at 32k depth |
| one user, fastest generation | `-c 32768 -np 1 --spec-type draft-mtp --spec-draft-n-max 2` | MTP: 60.6 / 56.8 t/s on two greedy prompts vs 37.7 without; draft 3: 58.2 / 52.7 |
| several requests (evaluation) | `-c 98304 -np 4` | 72 t/s total over 4 streams |
| f16 cache | drop `-ctk q8_0 -ctv q8_0` | the cache needs twice the memory per token, so lower `-c` accordingly (not measured); generation at short depth is about the same |

`--spec-type draft-mtp` uses the model's own multi-token-prediction layer as the draft model (no extra file);
`--spec-draft-n-max N` drafts up to N tokens per step, which the full model then checks in one batch. Output is the
same as without MTP up to rounding (`eval-results.md`). MTP was measured with one slot only; for several users, 4
plain slots give more total throughput (72 t/s) than one MTP stream (~60 t/s).

### Ternary Bonsai 27B (Bonsai fork)

```bash
/opt/llama.cpp-bonsai/build-gfx906/bin/llama-server -m /opt/llms/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -ngl 99 -fit off -fa on -ctk q8_0 -ctv q8_0 -c 253952 -np 1 --host 127.0.0.1 --port 8100 --no-webui \
    --reasoning-format deepseek
```

| use | change | measured |
|---|---|---|
| longest context | `-c 253952 -np 1` (PQ2_0) | largest context that fits; generation ~45.6 t/s at 32k depth |
| fastest generation | `-m /opt/llms/bonsai-mtp/Bonsai-2-27B-PQ2_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 2` | 68.8 / 64.8 t/s (two prompts) vs ~53 without. The MTP file is PQ2_0 plus the MTP layer |
| PTQ1_0 file | `-m /opt/llms/Ternary-Bonsai-2-27B-PTQ1_0.gguf` | same outputs as PQ2_0 (same ternary weights), smaller file, similar speed |
| evaluation | `-c 98304 -np 4` | |

Bonsai fork defaults to keep: the GDN fusions are on by default on gfx906 (`GGML_CUDA_GDN_FUSION=0` turns them off),
and PTQ1_0 prompts use the MMQ kernel instead of the fp16 hipBLAS path.

### Evaluation variants (`run_bench.sh`, `run_longctx.sh`)

All use the common command; only context and slots change:

| benchmark | `-c` | `-np` | per slot | why |
|---|---:|---:|---:|---|
| HumanEval+, MBPP+, LiveCodeBench, IFEval, MMLU-Pro, MATH-500, GSM8K | 98304 | 4 | 24,576 | up to 20,480 tokens per answer |
| AIME 2025 | 98304 | 3 | 32,768 | 24k thinking budget, 28,672 max tokens |
| LiveCodeBench at xhigh with 32k budget, Qwen | 81920 | 2 | 40,960 | 36,864 max tokens per answer |
| LiveCodeBench at xhigh with 32k budget, Bonsai | 163840 | 4 | 40,960 | Bonsai has room for 4 such slots |
| LongBench v2 | 104448 | 1 | 104,448 | documents cut to ~92k tokens + 4.6k answer |
| long-context retrieval at length L | L + 6144 (max 253952) | 1 | | room for thinking and the answer |
| greedy and MTP checks | 8192 | 1 | | `--spec-type draft-mtp --spec-draft-n-max 2/3` added for the MTP runs |

### Reference and stock runs

| run | command |
|---|---|
| Qwen3.8 Q6_K reference (22 GB, does not fit in 16 GB) | stock binary, `-m /opt/llms/ref/Qwen3.8-27B-UD-Q6_K.gguf`, no `-ngl` and `-fit` left on, so as many layers as fit go on the GPU and the rest run on the CPU (used with `llama-perplexity` for the KLD reference; `serve.sh qwen-q6` does the same for the server) |
| stock comparisons | the same commands with the stock binaries |
| stock PTQ1_0 workaround | `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` before the stock Bonsai command: hipBLAS computes in fp32 (KLD vs CPU 0.0035 -> 0.000066, prompts ~17% slower). Not needed with the Bonsai fork |

### Round 2 models (provisional; context limits and drafter settings are being measured)

| model | command (after the common options) | drafter |
|---|---|---|
| Gemma 4 26B-A4B | `-m /opt/llms/new/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf` | `-md /opt/llms/new/mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp --spec-draft-n-max N` |
| Gemma 4 31B | `-m /opt/llms/new/gemma-4-31B-it-Q3_K_S.gguf` | `-md /opt/llms/new/mtp-gemma-4-31B-it.gguf --spec-type draft-mtp --spec-draft-n-max N` |
| Muse Glimmer 30B | `-m /opt/llms/new/Muse-Glimmer-30B-UD-Q3_K_XL.gguf` | `-md /opt/llms/new/dflash-kquant.gguf --spec-type draft-dflash --spec-draft-n-max N` |

`-md FILE` loads a separate draft model. Gemma 4's drafter is a small "assistant" model that shares the main model's
cache (`draft-mtp`); Glimmer's is a DFlash block-diffusion drafter that proposes a whole block of tokens per step
(`draft-dflash`; `--spec-draft-n-max` is capped at its trained block size). This section gets the measured largest
context and the best draft length once the round 2 setup runs finish.

## Per-request options (chat completions API)

```bash
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 12288, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "seed": 12345,
  "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "medium"},
  "reasoning_budget_tokens": 8192
}'
```

| field | what it does |
|---|---|
| `chat_template_kwargs.enable_thinking` | thinking on / off (Qwen3.8, Bonsai, Gemma 4) |
| `chat_template_kwargs.reasoning_effort` | Qwen3.8 and Bonsai templates: `low`, `medium` (no extra instruction) or `xhigh` (template default; adds a "think carefully, validate assumptions" instruction to the system prompt) |
| `chat_template_kwargs.reasoning_strength` | Muse Glimmer: written into its system prompt as "Reasoning strength: X." (default `high`) |
| `reasoning_budget_tokens` | thinking budget for this request; when reached, the server inserts the budget message and closes the thinking |
| `max_tokens` | thinking + answer |
| `temperature`, `top_p`, `top_k`, `min_p` | sampling; Qwen3.8 and Bonsai recommend 1.0 / 0.95 / 20 (also in the GGUF metadata), Gemma 4 1.0 / 0.95 / 64 |
| `seed` | per-request sampling seed; the evaluation uses crc32(benchmark:question id) so all models get the same seed |
| `logprobs`, `top_logprobs` | return token probabilities (used by the MTP split check) |

## Fork switches (environment variables)

All default to the fork's faster path; set one to 1 to go back to stock behaviour for an A/B test.

| variable | effect when set |
|---|---|
| `GGML_CUDA_NO_FA_Q8_0_TILE=1` | the tile attention kernel stops reading q8_0 K/V directly and converts the cache to an f16 copy first (stock path; ~320 MiB more VRAM, slower generation at depth, ~4% faster prompts at 32k) |
| `GGML_CUDA_NO_MMVQ_HW=1` | disables the gfx906 multi-wave mat-vec kernel for IQ1/IQ2/IQ3/Q2_K (and IQ4_XS / Q3_K in some shapes) |
| `GGML_CUDA_NO_GDN_PAIR=1` | disables the fused alpha / beta mat-vec pair in the gated delta net layers |
| `GGML_CUDA_NO_NORM_Q8=1` | disables the fused rms_norm * mul + q8_1 quantize before mat-vec |
| `GGML_CUDA_GDN_FUSION=0` | (Bonsai fork) turns off the GDN conv-chain and alpha/beta fusions, on by default on gfx906 |
| `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=N` | (Bonsai fork) largest batch that uses the PTQ1_0 MMQ kernel; larger batches use the fp16 BLAS path |
