# gfx906 evaluation results (2026-10-29)

Plan, settings and changes: `eval-plan.md`. Results page (dashboard): https://claude.ai/artifact/BiowJKhqhqzjwrU8kNbDfA
(private until shared). Raw answers, scores and logs: `/opt/eval/results`; tools: `/opt/gfx906-tools/eval`.

Models, each on its gfx906 fork build, llama-server with q8_0 K/V, thinking on, temperature 1.0 / top-p 0.95 /
top-k 20, one answer per question, a fixed per-question seed shared by both models:

- `qwen-q3`: Qwen3.8-27B-UD-Q3_K_XL (12.2 GiB), branch `gfx906-qwen38`.
- `bonsai-pq2`: Ternary-Bonsai-2-27B-PQ2_0 (6.7 GiB), branch `gfx906-bonsai`. PTQ1_0 holds the same ternary
  weights and gives the same logits in the fork, so it was checked (Phase 1) but not benchmarked.

## Benchmarks

Score with 95% bootstrap interval. xhigh = `reasoning_effort=xhigh`, same budgets, except LiveCodeBench at xhigh with
a 32k thinking budget (16k at medium).

| benchmark | n | Qwen medium | Bonsai medium | Qwen xhigh | Bonsai xhigh |
|---|---:|---:|---:|---:|---:|
| HumanEval+ (plus) | 164 | 95.1 [91.5, 98.2] | 93.3 [89.0, 97.0] | 95.7 | 93.9 |
| MBPP+ (plus) | 378 | 81.0 [77.0, 84.7] | 79.4 [75.1, 83.3] | 80.7 | 81.0 |
| LiveCodeBench v6, newest 100 | 100 | 79.0 [71, 87] | 66.0 [57, 75] | 80.0 (32k) | 71.0 (32k) |
| IFEval, prompt strict | 270 | 89.3 [85.6, 92.6] | 88.1 [84.4, 91.9] | | |
| MMLU-Pro | 350 | 82.3 [78.3, 86.0] | 78.3 [74.0, 82.6] | | |
| MATH-500 sample | 200 | 96.0 [93.0, 98.5] | 96.5 [93.5, 99.0] | 96.0 | 96.5 |
| AIME 2025 | 30 | 80.0 [66.7, 93.3] | 70.0 [53.3, 86.7] | 96.7 | 90.0 |
| LongBench v2 short, half | 93 | 65.6 [54.8, 75.3] | 73.1 [63.4, 81.7] | | |

- Qwen is clearly ahead on LiveCodeBench (13 points at medium, 9 at xhigh) and ahead on AIME and MMLU-Pro; Bonsai is
  ahead on LongBench v2 (intervals overlap). HumanEval+, MBPP+, IFEval and MATH-500 are within about 2 points.
- LiveCodeBench problems are from Feb-Apr 2025 and the models are newer, so contamination is possible.
- xhigh vs medium: large gain on AIME (+17 / +20 points for ~20% more tokens); LiveCodeBench +1 (Qwen) / +5 (Bonsai)
  at 32k for ~1.7x tokens; HumanEval+, MBPP+, MATH-500 within 2 points while HumanEval+ / MBPP+ use 2.1-2.4x tokens.
  MATH-500 had the same score at both efforts by coincidence (Qwen 5 questions flipped each way, Bonsai 2).

## Long-context retrieval (own generator, novel passages with planted facts)

Correct / asked. Qwen up to 100k (its cache limit in this setup), Bonsai up to 250k.

| task | model | 8k | 32k | 64k | 100k | 128k | 196k | 250k |
|---|---|---|---|---|---|---|---|---|
| find a planted fact | Qwen | 60/60 | 40/40 | 30/30 | 20/20 | | | |
| | Bonsai | 60/60 | 40/40 | 30/30 | 20/20 | 20/20 | 10/10 | 10/10 |
| list every value of a key | Qwen | 18/18 | 11/12 | 9/9 | 6/6 | | | |
| | Bonsai | 17/18 | 10/12 | 8/9 | 5/6 | 2/6 | 3/3 | 0/3 |
| follow a chain of assignments | Qwen | 24/24 | 15/16 | 12/12 | 8/8 | | | |
| | Bonsai | 24/24 | 15/16 | 9/12 | 8/8 | 8/8 | 3/4 | 3/4 |
| most frequent items | Qwen | 4/6 | 3/4 | 3/3 | 2/2 | | | |
| | Bonsai | 3/6 | 1/4 | 2/3 | 0/2 | 0/2 | 0/1 | 0/1 |

## Fork correctness (Phase 1)

- Logits, mean KLD over prose / code / docs (llama-perplexity): fork vs stock, same ubatch and cache type
  - Qwen: prompt f16 0 (identical), prompt q8_0 0.0012, generation (ub 1) 0.0008-0.0009, MTP verify (ub 3)
    0.0008-0.0009. Noise floor: stock vs stock at ubatch 4 vs 1 is 0.0003, 512 vs 1 is 0.0007.
  - Bonsai PQ2_0: 0.00013-0.00014 everywhere. PTQ1_0: 0.00013-0.00015 for generation; at prompt size the fork matches
    the CPU reference (0.00014) where stock is off (0.0033, fp16 hipBLAS path, reported to PrismML).
  - vs the Q6_K reference (q8_0 K/V): Qwen Q3_K_XL 0.065 (code 0.14), Bonsai 0.40 (PQ2_0 and PTQ1_0 identical).
- Greedy text, 20 prompts x 512 tokens: MTP draft 2 and draft 3 give identical text (both models). Against no MTP, 9/20
  (Qwen) and 8/20 (Bonsai) replies are identical; every split is at a near-tie: the top-2 token probabilities at the
  split differ by at most 0.028 (Qwen, median 0.012) / 0.020 (Bonsai, median 0.006), and only 69 of 7,849 / 98 of 8,509
  generated tokens are that close. MTP verifies drafts in a 3-token batch whose arithmetic order differs from
  single-token generation, so it is lossless up to rounding, not bit-identical. Fork vs stock greedy text looks the
  same (Qwen 4/20 identical, Bonsai PQ2 14/20, median first split ~600-700 characters).

## Cost

GPU time for everything above, scoring included: ~37.7 h for Qwen, ~29.7 h for Bonsai (per-run hours on the results
page). The largest items were LiveCodeBench at xhigh / 32k (10.5 h / 6.1 h), LongBench v2 (4.5 h / 3.9 h) and
LiveCodeBench at medium (4.5 h / 3.0 h).

## Notes

- Sample cuts made for time (`eval-plan.md`): PTQ1_0 not benchmarked, GSM8K dropped, IFEval and MMLU-Pro halved
  (seeded). GPQA not run (gated dataset).
- LongBench v2: one document (a numeric weather CSV, ~1 character per token) was first cut by a 4-chars-per-token
  estimate and rejected as too long; it is now cut with its measured ratio and was rerun for both models (`prep_longbench.py`).
