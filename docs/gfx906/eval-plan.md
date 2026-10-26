# Evaluation plan: Qwen3.8 27B Q3_K_XL vs Ternary Bonsai 2 27B (gfx906, Radeon VII)

Status: approved 2026-10-26 with the default choices; running. Implementation notes and changes made while setting
it up are in "Implementation" at the end.

Two questions:

1. **Did any of our speedups change model outputs?** (regression)
2. **How good are the three quantized models, relative to each other and to a high-precision reference?**
   Priority: coding first, then chat / instruction following, then long documents.

## Models

| id | file | size | build | role |
|---|---|---:|---|---|
| `qwen-q3` | Qwen3.8-27B-UD-Q3_K_XL | 12.2 GiB | our fork, `gfx906-qwen38` @ `6b2f33d` | under test |
| `bonsai-pq2` | Ternary-Bonsai-2-27B-PQ2_0 | 6.7 GiB | PrismML fork, `gfx906-bonsai` @ `c389227e2` | under test |
| `bonsai-ptq1` | Ternary-Bonsai-2-27B-PTQ1_0 | 5.5 GiB | same | under test |
| `qwen-q6` | Qwen3.8-27B-UD-Q6_K (unsloth) | 22.0 GB | stock upstream `f5027fb` (`/opt/llama.cpp-stock`) | reference |

- The Bonsai MTP file (`Bonsai-2-27B-PQ2_0-MTP.gguf`) has byte-identical main weights to `bonsai-pq2`, so it is not
  benchmarked separately; only its MTP output equivalence is checked (Phase 1).
- `qwen-q6` does not fit in 16 GB VRAM: it runs partly on the CPU (about 16 of 22 GB on the GPU), so it is slow
  (estimate ~8-10 t/s). It is used for logit comparisons (KLD) only, plus optional task runs (Phase 2, tier 3).
- The reference runs on the **stock** upstream build, so it is independent of our kernel changes.

## Serving setup (same for all models)

- llama-server, `-ngl 99 -fa on -fit off`, **q8_0 K/V cache** (the setting that large contexts need; validated in the
  long-context round), `-np 4` (4 parallel slots: 72 t/s aggregate for `qwen-q3` vs 35.5 for 1 slot; 8 slots drop to
  52 t/s), chat completions API with each model's own chat template.
- Short-form benchmarks: 4 slots x 24k tokens of context each.
- Long-context tests: 1 slot, context as large as the test needs (`qwen-q3` up to 106k, Bonsai up to 254k).
- No MTP during benchmarks: MTP does not change greedy output (checked in Phase 1), and 4 parallel slots give more
  throughput than 1 slot with MTP (72 vs ~60 t/s).
- **Sampling:** the models' recommended settings from their GGUF metadata (temperature 1.0, top-p 0.95, top-k 20),
  with a fixed per-question seed, identical for all models. Thinking models can loop under greedy decoding, so greedy is
  used only for the regression checks. Every score gets a 95% confidence interval (bootstrap over questions).
- **Thinking mode** (`enable_thinking`, supported by both templates):
  - on for coding, maths and GPQA, with a per-benchmark token cap (below);
  - off for IFEval, MMLU-Pro and the long-context tests, where answers are short and thinking would multiply the cost.
  - Answers cut off by the cap count as wrong, and the number of capped answers is reported per model.
- Every run logs tokens generated, wall time and t/s per request, so speed and quality are reported together.

## Phase 0: setup and pilot (~2-3 h)

- Install the harnesses in a venv (`/opt/eval/venv`). Per AGENTS.md, downloaded code is read before it runs; any
  Python run directly from a download uses `python3 -I`.
  - lm-evaluation-harness (EleutherAI): IFEval, MMLU-Pro, GPQA Diamond, MATH-500, GSM8K, through its
    `local-chat-completions` backend.
  - EvalPlus: HumanEval+ and MBPP+.
  - LiveCodeBench (official runner, OpenAI-compatible backend).
  - LongBench v2 (THUDM dataset; small own client).
  - Long-context retrieval: our own RULER-style generator (stdlib Python, in `/opt/gfx906-tools/eval/`), so every
    prompt and score is easy to audit.
- **Model-written code execution (EvalPlus, LiveCodeBench)** runs through `/opt/gfx906-tools/eval/sandbox.sh`
  (the whole evaluation step, not only single tests):
  - The container is an apptainer `--fakeroot` instance: container root is host uid 1008 (the user's account), and
    container uids 1-65536 map to the host subordinate range starting at 624288. So container uid 1000 is host uid
    625287, which owns nothing on the host.
  - `sandbox.sh <workdir> <timeout> <cmd>`: new network namespace (loopback only, so llama-server and the internet
    are unreachable) and a private mount namespace in which `/tmp` (the host's `/tmp` is bind-mounted here), `/opt`,
    `/home`, `/root` and `/data` are replaced by empty tmpfs; only the work dir (under `/opt/eval/sbx/`) is visible,
    at `/run/sbxw`.
  - Then `setpriv` to uid/gid 1000, no capabilities (inheritable and bounding sets cleared), `no_new_privs`, and
    `prlimit` limits (4 GiB address space, 256 processes, 256 MiB file size, 256 open files) plus a wall-clock timeout.
  - Tested 2026-10-26: no network (`ENETUNREACH`), models and repos invisible, `/dev/kfd` and `/etc/shadow`
    unreadable, `/usr` read-only, a 6 GiB allocation fails, an infinite loop is killed at the timeout.
  - Remaining risk: a kernel exploit from inside the namespaces. Acceptable for benchmark code that is not
    adversarial.
- **Pilot:** 10 questions of every benchmark on `qwen-q3`, to check that answer extraction and scoring work, and to
  measure the real tokens per answer. The sample sizes below are then scaled to the time budget (the pilot numbers and
  any changes get written into this file before the full run).

## Phase 1: regression and reference checks (~4 h)

### 1a. Our builds vs their stock versions (did the speedups change outputs?)

`llama-perplexity --kl-divergence`, same text, logits of our build vs the stock build of the same model:

| model | stock build | ubatch | K/V cache |
|---|---|---|---|
| `qwen-q3` | upstream `f5027fb` | 1, 3, 512 | f16 and q8_0 |
| `bonsai-pq2`, `bonsai-ptq1` | PrismML `prism` @ `97ea5aeee` | 1, 3, 512 | f16 and q8_0 |

- ubatch 1 = generation kernels, 3 = MTP verify, 512 = prompt.
- Reference point: stock compared with itself at a different ubatch size (e.g. stock ubatch 4 vs 1: KLD 0.00031 for
  `qwen-q3`). Differences of that size are summation order, not damage.
- Known exception: stock PTQ1_0 at ubatch 512 is itself 30x off (the fp16 hipBLAS bug reported to PrismML); for that
  case our build is compared with the CPU logits instead.
- Three texts, 16 chunks of 512 tokens each: English prose (wikitext-2 test), Python source code (CPython stdlib from
  `/usr/lib/python3.12`), and technical documentation (the current `kld-text.txt`).

### 1b. All models vs the Q6_K reference (quality proxy)

- KLD, perplexity ratio and same-top-token rate of each model against `qwen-q6`, on the same three texts.
- Bonsai is a retrained ternary / 2-bit derivative of Qwen3.8 27B, so "how close to the full model" is a fair first
  measure. It is not the final word: the task benchmarks decide.

### 1c. Generation-level checks

- **MTP is lossless:** greedy, 20 prompts x 512 tokens; output with MTP (draft 2 and 3) must equal output without
  MTP, for `qwen-q3` and the Bonsai MTP model.
- **Fork vs stock greedy text:** the same 20 prompts on our build and the stock build. The point where the two first
  differ is reported (rounding makes them drift apart eventually; an early divergence on many prompts would be a red
  flag).
- **q8_0 vs f16 K/V:** KLD on the three texts, so the cost of the cache setting used in all benchmarks is known.

## Phase 2: short-form benchmarks (per model, ~6-9 h for `qwen-q3`, less for Bonsai)

Run benchmark by benchmark across all three models (not model by model), so every finished benchmark is a complete
comparison even if the run is stopped early. Order = priority.

| tier | area | benchmark | questions | thinking | token cap | est. tokens / model |
|---|---|---|---:|---|---:|---:|
| 1 | coding | HumanEval+ (EvalPlus) | 164 (all) | on | 8k | ~0.3M |
| 1 | coding | MBPP+ (EvalPlus) | 378 (all) | on | 8k | ~0.5M |
| 1 | coding | LiveCodeBench, newest release window | ~100 (most recent) | on | 16k | ~0.6M |
| 1 | chat | IFEval | 541 (all) | off | 2k | ~0.2M |
| 2 | knowledge | MMLU-Pro, stratified sample | 1,400 (100 x 14 subjects) | off | 2k | ~0.4M |
| 2 | reasoning | GPQA Diamond | 198 (all) | on | 12k | ~0.8M |
| 2 | maths | MATH-500, fixed random sample | 200 | on | 8k | ~0.6M |
| 2 | maths | GSM8K, fixed random sample | 250 | off | 2k | ~0.1M |
| 3 | maths | AIME 2025 | 30 | on | 24k | ~0.5M |
| 3 | reference | HumanEval+ and IFEval on `qwen-q6` | | as above | | (slow, ~12 h) |

- **Scoring:** HumanEval+ / MBPP+ pass@1 on the "plus" tests (base tests reported too); LiveCodeBench pass@1 by
  difficulty; IFEval prompt- and instruction-level strict accuracy; MMLU-Pro / GPQA accuracy (MMLU-Pro also per
  subject); MATH / GSM8K / AIME exact match after answer normalisation.
- **Token estimates** are guesses until the pilot. At ~72 t/s (`qwen-q3`, 4 slots), tier 1 is ~6 h and tier 2 another
  ~7 h; Bonsai should be faster (to be measured). If the pilot shows the totals far over budget, tier 2 sample sizes are
  cut first, tier 1 stays complete.
- **Chat beyond IFEval** needs a judge model (MT-Bench, Arena-Hard). Optional: I can judge a blind, randomised sample
  myself (e.g. MT-Bench's 80 two-turn questions, pairwise, model names hidden). That costs a few hundred thousand of my
  tokens, so it is listed but not planned unless you want it.

## Phase 3: long-context tests (per model ~3-4 h, Bonsai long lengths extra)

### 3a. Retrieval (RULER-style, own generator)

- Haystack: real prose (public-domain books from Project Gutenberg), so the filler is natural text.
- Tasks:
  - **single needle** (one key-value fact, find its value),
  - **multi-key** (several similar keys, distractors, find the right one),
  - **multi-value** (one key, several values, return all),
  - **variable tracking** (chains of assignments `X1 = 12345, X2 = X1, ...`, return all names with a given value),
  - **frequent-word aggregation** (return the 10 most frequent of a word list).
- Lengths (tokens): 8k, 32k, 64k, 100k for all models; 128k, 192k, 250k for Bonsai only (`qwen-q3` does not fit them).
- Cost control: prefill dominates (100k tokens is ~11 min at depth). Each long document carries 10 needles at spread
  depths, and the questions about them reuse the cached document (llama-server prompt cache), so one prefill gives 10
  scored answers. 2 documents per task and length: 20 answers per cell.
- Scoring: exact match per answer (all values for multi-value / variable tracking), reported as accuracy per task and
  length, plus accuracy by needle depth.

### 3b. Long documents (LongBench v2)

- Multiple-choice questions over real long documents (QA, multi-document QA, code repositories, long dialogue,
  structured data).
- The "short" subset (documents under ~32k words, ~180 questions); questions that do not fit a model's context are
  truncated in the middle as the LongBench v2 protocol specifies. If time allows, the "medium" subset for Bonsai only
  (fits its 254k context).
- Thinking off, answer letter extracted; accuracy overall and per category, with random-guess baseline 25%.

## Phase 4: report

- One results page: all scores with confidence intervals, tokens and t/s per benchmark, per model. Also: tables and
  charts for the long-context accuracy by length, plus the Phase 1 regression tables.
- Raw outputs (every prompt, response and score) kept in `/opt/eval/results/<benchmark>/<model>/` for spot checks.
- Methods and results also written into `docs/gfx906/` (both forks) as usual.

## Time budget (estimates, to be replaced by pilot measurements)

| phase | `qwen-q3` | each Bonsai | total |
|---|---:|---:|---:|
| 0 setup + pilot | | | 2-3 h |
| 1 regression + reference | | | ~4 h |
| 2 tier 1 (coding + IFEval) | ~6 h | ~4-5 h | ~15 h |
| 2 tier 2 (knowledge, reasoning, maths) | ~7 h | ~5 h | ~17 h |
| 3 long context | ~3-4 h | ~4-6 h (incl. 128k-250k) | ~13 h |

- Tiers 1 + Phase 3 plus Phases 0, 1 and 4: about 1.5 days of GPU time. Adding tier 2 brings it to about 2 days.
- Runs are unattended, resumable (each answer is saved as it finishes), and go through `gpu.sh` like all GPU jobs.

## Open choices (defaults in bold)

- Sampling: **recommended settings + fixed seeds** / greedy.
- K/V cache for benchmarks: **q8_0** / f16 (f16 limits `qwen-q3` to ~57k total context, i.e. 4 slots x 14k).
- Tier 2: **run after tier 1 and Phase 3** / skip / run before Phase 3.
- Judge-based chat evaluation (MT-Bench, judged by me): **not planned** / add.
- Q6_K reference on task benchmarks (tier 3): **not planned** / add (~12 h).

## Implementation (2026-10-26)

Code: `/opt/gfx906-tools/eval/` (not in git, like the other tools). Data: `/opt/eval/data/`, results:
`/opt/eval/results/<benchmark>/<model>/{gen.jsonl, score.json}`, server logs: `/opt/eval/logs/`.

Installed for this (also listed in README "Packages installed"): apt `python3.12-venv`, `python3-pip`; venv
`/opt/eval/venv` with evalplus 0.3.1, math-verify 0.9.0, huggingface_hub, pyarrow, langdetect, nltk, absl-py,
immutabledict, numpy, tqdm. Downloaded code: LiveCodeBench repo @ `28fef95` (only its 3 evaluation files are used),
google-research `instruction_following_eval` @ `e6890f85` (4 files, read: text checks only). NLTK punkt data in
`/opt/eval/nltk_data` (NLTK refuses group-writable dirs: `/opt/eval` is now `g-w`).

| script | what |
|---|---|
| `serve.sh <model-id> <port> <ctx> <np> [args]` | llama-server for one model (q8_0 K/V, `--reasoning-format deepseek`, budget message), waits for ready |
| `gen.py` | generation client: chat completions, N in flight, per-question seed (crc32 of benchmark:id), resumable JSONL |
| `run_bench.sh <bench> <model-id> [limit]` | GPU lock + server + `gen.py` with that benchmark's settings |
| `run_longctx.sh <model-id> <length>` | same for the long-context set (1 slot, ctx = length + 4096) |
| `sandbox.sh` | code execution sandbox (see Phase 0) |
| `prep_*.py` | prompts + gold answers per benchmark |
| `score_evalplus.sh`, `score_lcb.sh` | scoring in the sandbox; `score_ifeval.py`, `score_simple.py`, `score_longctx.py` outside |
| `phase1-kld.sh`, `phase1-greedy.sh`, `greedy-compare.py` | Phase 1 |
| `summarize.py` | all scores with bootstrap 95% CIs, budget hits, tokens |

Decisions and changes against the plan above:

- **Thinking budget instead of a hard cap:** llama-server's per-request `reasoning_budget_tokens` ends the thinking
  block at the budget and injects "Considering the limited time by the user, I have to give the solution based on the
  thinking directly now." (the wording from the Qwen3 report), then the model writes its answer. `max_tokens` = budget
  + 2-4k for the answer. Answers that hit the budget are counted per model (`budget_hit`); answers that still hit
  `max_tokens` count as wrong (`cut`).
- **Parallel slots:** 4 slots, unified K/V of 98,304 tokens (q8_0), so one long answer can use more than a quarter.
  AIME uses 3 slots.
- **LiveCodeBench:** the newest official release is v6 (problems Jan-Apr 2025, 175). Used: the 100 most recent
  (2025-02-15 .. 2025-04). Qwen3.8 is newer than these problems, so contamination is possible; it affects all three
  models alike (same base model), so the comparison holds but absolute scores may be flattering. The private tests are
  pickled in the dataset: decoded with an unpickler that refuses any global (plain strings only); all 175 decode.
- **GPQA Diamond: skipped for now (owner's decision, 2026-10-26), possible later addition.** It is gated on Hugging
  Face (`Idavidrein/gpqa`: accept the terms, then a read-only token). Ready to run once the CSV is in
  `/opt/eval/data/gpqa/`: `prep_gpqa.py` (choices shuffled with a fixed seed), then `run_all.sh gpqa`
  (`run_bench.sh` already has its settings: thinking on, budget 12k). `run_all.sh` skips it while the data is missing.
- **IFEval** has 541 prompts (the file's last line has no newline).
- **MMLU-Pro** zero-shot with the official "the answer is (X)" instruction; extraction as in the official script
  (answer-is pattern, then "Answer: X", then the last standalone letter); no random guess when nothing is found.
- **Maths** (MATH-500, GSM8K, AIME): "Please reason step by step, and put your final answer within \boxed{}.";
  scored with `math-verify` (checked: `\dfrac` vs `\frac`, tuples, plain numbers).
- **Long-context retrieval (3a), as built:** each document is a passage from Project Gutenberg novels (War and Peace,
  Les Miserables, Moby Dick, Great Expectations, A Tale of Two Cities, Pride and Prejudice) with 65 inserted sentences;
  18 questions per document: 10 single-value keys at depths 5%..95% (each with 12 other keys of the same form as
  distractors), 3 four-value keys, 4 variable chains of 4 hops, and 1 aggregation question (most and second most
  frequent of 8 items in 27 "Inventory log" lines; replaces the frequent-words task, so it can share the document).
  Documents per length: 8k: 6, 32k: 4, 64k: 3, 100k: 2, 128k: 2, 192k: 1, 250k: 1 (the last three Bonsai only).
  Actual prompt lengths are ~6% below the nominal ones (30,948 tokens for "32k"); reports use the actual counts.
  Prompt cache reuse works on the hybrid model (checked): the first question of a 31k document took 101 s, each
  next one 2.6 s (~516 tokens re-processed from the last checkpoint).
- **LongBench v2 (3b):** moved to `zai-org/LongBench-v2`. The short subset needs ~5.4M prompt tokens per model
  (~5-6 h each), so a stratified half is used: 93 questions (half of each domain x difficulty cell, seed 1234),
  documents over ~92k tokens cut in the middle (2 of 93). Run shortest first.
- **MTP check (1c):** the MTP verify step runs 3-token kernels, so its logits are not bit-identical to single-token
  decoding; greedy text can differ after a near-tie. Reported as identical prompts and first differing character, like
  the fork-vs-stock comparison.
