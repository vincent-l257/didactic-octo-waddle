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

## Results (IQ3_S, 18k prompt, one run each)
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

Split sweep: 9-10 / 28-29 are flat (~1380); wall 12.5 s vs V100 9.4 s busy, so ~3 s is pipeline fill/drain.
Best now: `--layer-split 10,28` (serve3_iq3_1028.bat): beats two cards (1301 / 68.0) on both prompt and decode.

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
1d. Pipeline fill (~3 s of 12.5): a short first chunk so the V100 starts sooner.
2. (next) Experts on the V100 at ~17 TOPS of dp4a (~60% of peak): next step is int8 -> fp16 in registers and m8n8k4 HMMA
   (exact: per-32 integer sums < 2^24), several times the dp4a rate.
3. gdn_chunked on WMMA/FP16.
4. (done) drafter-row report made async.
5. Rebalance layer split once the V100 is faster (more layers to it).
6. Decode kernels on the V100 (qsa_dec_attn, head).

## Not to do
- Don't push anywhere. Quality check (PPL/KLD) before keeping numerics changes as default.
