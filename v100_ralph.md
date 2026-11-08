# V100 loop notes

Goal: find and do everything that makes the V100 (sm_70, CUDA 12.8 cubins in build-v100/bin/sm70) add speed to the
three-card IQ3_S setup. Each iteration: profile, fix the largest V100 cost, rebuild, measure, record here.

## How to measure
- Build: `cmd //c <scratchpad>/build-v100.bat` (stop the test server first: it locks the exe).
- Server: `<scratchpad>/serve3_iq3_prof.bat` (BASALT_PF_PROF=1, port 8096, layer split 8,30, V100 = CUDA2 last stage).
  Without the profiler: `serve3_iq3.bat` (profiler syncs per layer, so it reads slightly slow).
- Prompt: `curl -d @<scratchpad>/long.json http://127.0.0.1:8096/v1/chat/completions` (18,096 tokens, greedy, 200 out).
- Read `~/.cache/basalt/9ee2c74ced041f2c/engine.log`: `pf-prof: CUDA2 ...` (ms per part over 64 layers) and
  `prompt stages: ...` (busy per card, hand-off, report).
- Greedy identity: compare the reply with the previous run's json in the scratchpad.

## Results (IQ3_S, 18k prompt, 32k context, one run each)
| change | prompt tok/s | decode tok/s | V100 busy |
|---|---|---|---|
| two cards only | 1301 | 68.0 | - |
| WMMA FP16 GEMM | 795 | 81.1 | 19.06 s |
| + dp4a expert dots (identical output) | 853 | 83.2 | 17.64 s |
| + hc up-mix via GEMM on sm_70 (FP16 rounding of w_up; output changes slightly) | 977 (profiled) | 79.3 | 14.90 s |
| + Strata's m8n8k4 prompt attention (launch70; output changes slightly) | 1015 (profiled) | 79.6 | 13.82 s |

| layer split 8,28 instead of 8,30 (V100 20 layers; plain run) | 1153 | 84.8 | 12.98 s |
| layer split 8,26 (V100 22 layers: V100 is the bottleneck again) | 1087 | 87.2 | 13.86 s |
| + async drafter report at 8,28 (report 2242 -> 38 ms; output identical to sync) | **1231** | **91.2** | 11.69 s |

| split 9,28 (async report) | 1377 | 95.8 | 9.43 s |
| **split 10,28** | **1382** | **99.1** | 9.48 s |
| split 10,29 | 1387 | 95.9 | 9.27 s |
| split 10,28, --prefill 4096 (worse: keep 8192) | 1296 | 82.7 | 11.36 s |

4096-chunk decode drop (82.7 vs 99.1) is not drafts (121/172 vs 122/179 accepted): the server chose
resident=7.0GB with 4096 chunks vs 11.2GB with 8192, so expert placement differs. Not dug into.
Split sweep: 9-10 / 28-29 are flat (~1380); wall 12.5 s vs V100 9.4 s busy, so ~3 s is pipeline fill/drain.
Best now: `--layer-split 10,28` (serve3_iq3_1028.bat): beats two cards (1301 / 68.0) on both prompt and decode.

## At 262k context (the real setting; one run each)
Every table below this one ran with `--max-context 32768` (a launcher override from the first three-card test,
copied into every later launcher). basalt_config.json sets 262144; launchers now carry no override.
| model, split | 18k prompt: prompt / decode tok/s | 125k prompt: prompt / decode tok/s |
|---|---|---|
| IQ3_S 10,28 | 1297 / 97.3 | 1802 / 89.4 |
| IQ4_XS 10,30 | 1078 / 67.0 | 1580 / 65.0 |
Both keep the V100's own prompt buffers (no loan) at 262k. IQ3_S: 100% of experts in VRAM; IQ4_XS: 92%.
Long prompt: `<scratchpad>/huge.json` (docs + src, 125,278 tokens); `PROMPT=huge measure.sh <tag> <log>`.

## Decode profile (IQ3_S 10,28, 262k, 18k prompt, 400 out; `--profile`, every stage now reported)
verify 22.87 ms/window (2.30 tokens), draft 2.43. GPU ms/window per stage, run one after another:
- CUDA0 (5070 Ti, 10 layers): 2.86 (0.29/layer)
- CUDA1 (3090, 18 layers): 6.15 (0.34/layer)
- CUDA2 (V100, 20 layers): 12.80 (0.64/layer). Biggest parts: expert VRAM hits 2.96, hc-read1+router 2.12,
  hc-read0 1.60, q8+qkv gemv 1.18 (+0.45 QSA q), shared+quant 0.91, head 0.83, out-proj 0.68, attention 0.47.
The head is 0.83 ms: not the problem. The V100 is ~2x the 3090 per layer at similar memory bandwidth, across
the memory-bound kernels (hc reads, gemv, expert hits). If it matched the 3090 per layer (~6.8 ms), verify would
drop ~6 ms (estimate, not measured). Next: find why fused_gr / native_mmvq / expert hits run ~2x on sm_70.

## IQ4_XS on three cards (same build, 18k prompt, 32k context, one run each)
| setup | prompt tok/s | decode tok/s | V100 busy |
|---|---|---|---|
| two cards (earlier) | 992 | 55.0 | - |
| three cards --prompt-dup (earlier) | 764 | 64.4 | - |
| split 10,28 (V100 20 layers) | 394 | 62.7 | 32.3 s |
| **split 10,30** (V100 18 layers; serve3_iq4_1030.bat) | **1084** | **73.6** | 10.9 s |
| split 10,31 (more experts in RAM: decode drops) | 1079 | 69.9 | 10.8 s |

At 10,28 the V100 lacks room for its own prompt buffers, borrows 1626 cache slots (3.78 GiB) and streams 3577
experts back over x1 per prompt (~8.6 GB at 0.4 GB/s). Rule: the V100 must keep its own buffers ("CUDA2 keeps its
own prompt buffers ... no loan" in engine.log); pick the split from that line. At 10,30: 94% of experts in VRAM,
6% in RAM; stages CUDA0 8.1 s, CUDA1 10.2 s (7619 streamed), CUDA2 10.9 s.

Async report: src/program/serve/async_report.hpp. The last stage's rows go D2D to a spare buffer, then D2H on a
side stream to pinned host (2 slots); a worker thread runs the drafter's prefill in order. Prefill::on_drain
(pipeline.cpp drain_pipeline) waits for it at the end of every run. The relay copy in mtp/prefill.cpp is now
cudaMemcpyDefault, so host rows work. Stages at 8,28: CUDA0 10.6 s, CUDA1 11.8 s (5158 streamed), CUDA2 11.7 s.

After the attention port the 3090 (CUDA1, 15.05 s busy, 6488 streamed) is the slowest stage.
V100 fine profile (64 layers, ms): between 3055 (idle: waits on CUDA1) hc 737 gdn.proj 458 gdn.rec 599 gdn.conv 160
gdn.out 133 qsa.attn 993 qsa.qkv 159 qsa.score 115 qsa.out 119 moe.route 406 moe.shared 223 moe.experts 3096 write 494.
Tools: `pf_tick(stream, "label")` / `pf_report(64)` in src/prefill/pf_prof.{hpp,cpp}; `<scratchpad>/measure.sh <tag> <log>`.

## V100 profile after hc fix (CUDA2, 64 layers, ms)
hc 384, gdn 1161, qsa 3564, moe 3189, write 145. 3090 (CUDA1) for comparison: hc 419 gdn 596 qsa 875 moe 1474.
Drafter-row report over x1: 2.24 s.

## Still emulated on sm_70 (sm70_emu.cuh)
- qsa_prompt_attn.cu (m16n8k16 bf16 mma) - Strata has prompt_attn_v70_kernel (m8n8k4) to port
- native_qsa_score.cu (TF32 scorer), qsa_select.cu
- gdn_chunked.cu (bf16 mma)
- qsa_dec_attn.cu, q6k_head.cu (decode side)
- moe: dp4a now; remaining cost is routing/gathers/quantize? to split

## Ideas queue
1. (done) qsa split; attention ported.
1b. (done) rebalanced: 8,28 best for prompts. Retry 8,27 / 9,28 once things shift again.
1c. User idea: read prompts on the two fast cards only (balanced cut, as the two-card run that got 1301), then hand
    the V100 layers' state (KV ~70 MB + GDN state for 18k) to the V100 for three-card decode. --prompt-dup did
    this unbalanced (5070 Ti 8 layers, 3090 40) and got 707. Make the dup prompt cut balanced and compare with 1231.
    Parked: the serve prompt split (prompt_split.cpp) is for one decode card plus helpers, not layer-split decode,
    so this needs new plumbing; its ceiling is about the two-card rate (~1300), since every chunk streams the V100
    layers' experts (~16 GB IQ3_S) from RAM into the 3090 over Gen3 x8. The three-card pipeline streams none of it.
1d. (tried, dropped) short first chunk for layer splits (BASALT_SPLIT_HEAD, prompt.cpp set_head_chunk): 2048 ->
    1387 tok/s vs ~1370-1382 without (noise), more experts streamed. Reverted.
2. (tried, parked) m8n8k4 FP16 expert dot (v100_hmma_experts.patch): identical output, no gain (V100 9.56 s vs
    9.48 s dp4a). First version spilled (255 regs, up to 648 B stack); fixed by one row tile of accumulators at a
    time. The products are bound by tile loading (IQ3_S decode in load_tiles), which both paths share.
2b. V100 expert sub-steps at 10,28 (60 layers, inflated by per-group syncs): x.gu 2108 x.down 1200 x.gather 838
    x.act 796 (swiglu + H quantize) x.pre 157 x.quant 65 ms. Others: qsa.attn 1109 gdn.rec 778 gdn.proj 578 hc 606
    between 1415. Next targets: IQ3_S load_tiles on sm_70, the resident-expert gather (copies every expert into
    group slots each chunk: ~14 ms per layer-chunk vs ~2.4 ms of bandwidth), gdn_chunked (emulated bf16 mma).
3. Quality: PPL/KLD of the V100 changes (WMMA FP16 GEMM, hc up-mix, m8n8k4 attention) - not run yet. Blocked:
   no BF16 .kld base (llama-perplexity --kl-divergence-base from the BF16 GGUF) and no wiki.test.raw on this
   machine (docs/precision/quality.md).

Loop stopped (2026-11-07): the rest are each <= ~5% of the V100's 9.5 s.
- x.gather is mostly profiler syncs (~1900 ticks); the real copy is ~3 ms per layer-chunk (~0.2-0.3 s total).
- iq3s_grid is __device__ (global, L1-cached), not constant memory: not the tile-loader cost.
- gdn: Strata's V100 path (src/prefill/gdn_chunk.cu: cumsum/kkt/wmma kernels) has a different structure from
  Basalt's gdn_chunked.cu; a port is a rewrite for <= ~0.5 s. Strata also has src/split/src/moe_mma.cuh
  (m8n8k4 expert products, 64-row jobs) - a reference if the expert tile loader gets reworked.
(old) Experts on the V100 at ~17 TOPS of dp4a (~60% of peak): next step is int8 -> fp16 in registers and m8n8k4 HMMA
   (exact: per-32 integer sums < 2^24), several times the dp4a rate.
3. gdn_chunked on WMMA/FP16.
4. (done) drafter-row report made async.
5. Rebalance layer split once the V100 is faster (more layers to it).
6. Decode kernels on the V100 (qsa_dec_attn, head).

## Not to do
- Don't push anywhere. Quality check (PPL/KLD) before keeping numerics changes as default.
