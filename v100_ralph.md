# V100 loop notes

Goal: find and do everything that makes the V100 (sm_70, CUDA 12.8 cubins in build-v100/bin/sm70) add speed to the
three-card IQ3_S setup. Each iteration: profile, fix the largest V100 cost, rebuild, measure, record here.

## How to measure
The full method is in `v100_howto.md`: the machine and the layer split (with each V100 layer's expert types), the
sm_70 build, the launchers and scripts, standalone kernel benches, and how the V100 kernels were written and wired in.

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

## Deep dive (2026-11-08): sFNX (github.com/johnfink8/sfnx, V100-native, same model) + per-kernel benches
Tools: nsys hangs the engine on this box (serve and --serve stdin alike): don't use it. Per-kernel truth comes from
standalone benches in `<scratchpad>/gt/` (kernels copied from src, built by `bh.bat <name>` with CUDA 12.8 for
sm_70/86/120). Engine-alone driver: `<scratchpad>/eng.sh` (GEN/GENT on stdin, QUIT after DONE).
Measured: graph node gap V100 2.2 us vs 5070 Ti 0.8 us (~50-90 us/layer of the V100's 640). V100 decode clocks
1380 MHz (its max), 97 W; 3090 ~1770 MHz. Dead ends: fused_gr plain/split/staged variants (decode runs hc8_gr, the
variants are dead code on 3090/V100); native_mmvq upstream layout (no faster: 1.25 vs 1.18 ms).
- **Done (11-09):** hc8 read as two kernels: split-K down (1280-column slices, norm folded in, rs per slice), up with a
  thread per half row and its weights loaded before lo is finished. One read at T=3, us old/new (`<scratchpad>/gt/hc8d.cu`):
  V100 55.9/33.9, 3090 40.8/25.2, 5070 Ti 31.3/14.6. IQ4_XS 10,29 long400, profiled: V100 stage 11.69 -> 10.85 ms, 3090
  7.29 -> 7.03, verify 23.81 -> 23.02, 84.2 -> 84.8 tok/s. Numerics: sum order only; the reply diverges at char 91.
- Draft threshold A/B (BASALT_DRAFT_TH temp override, IQ3_S 10,28, 262k, long400, no profiler; same greedy text):
  default 0.35: 102.1 tok/s (T 3.24, 2.46 tok/window, verify 21.66 ms); 0.15: 96.3 (T 4.01, 2.60, 24.10);
  0.5: 101.5 (T 3.13, 2.41, 21.30). A row costs ~3 ms here (10 more experts per token), not ~0: default stays.
- native_mmvq alone (`<scratchpad>/gt/mmb.cu`), T=3: V100 Q8_0 575-654 GB/s, Q6_K 400-465 GB/s; 3090 Q8_0
  776-818, Q6_K 483-640. V100 ~1.3x the 3090 (about the clock ratio), so queue item 1 is ~0.5 ms, not ~1 ms.
- expert_fp16.cpp (dequant to fp16 + WMMA GEMM per expert) is not an option: ~160-row GEMMs fill ~20 of 80 SMs.
  A grouped fp16-HMMA kernel (sFNX k_moe_mma style) is being written standalone in `<scratchpad>/gt/moe70/`.
- **Done:** prompt experts on f16 tensor cores on cc 7.0 (src/prefill/moe_v70.cu, from `<scratchpad>/gt/moe70/`;
  standalone at 81,920 rows: IQ3_S gate|up+swiglu 14.9 ms / 36 TFLOPS, IQ4_NL down 7.7 ms / 35 TFLOPS, ggml layout;
  repacked layouts 11.9 / 6.5 ms; ~1e-3 rel. error vs a double reference). Wired into compute_mmq per group
  (v70_l: f16 rows from mixed_h into GU, job list per group, fused swiglu H f16 into H, down into Dm).
  The pack is mixed-quant: V100 layers 28-47 gate|up are IQ3_XXS x9, IQ2_S x4, IQ3_S x6, IQ4_XS x1 (down IQ4_NL,
  Q2_0 on 30 and 36), so only 6 layers take it so far: prompt 1371 -> 1417 tok/s, decode 104.1, V100 busy 9.08 s.
  Next: IQ3_XXS / IQ2_S / IQ4_XS tiles (agent), then the repacked layouts.
- **Done:** IQ3_XXS, IQ2_S and IQ4_XS gate|up tiles (standalone, skewed 81,920 rows, fused swiglu: 13.4 / 13.5 /
  12.3 ms, 40-44 TFLOPS; ~1.3e-3 rel. error). 18 of the V100's 20 layers now take the f16 path (30 and 36 keep MMQ:
  Q2_0 down). IQ3_S 10,28, 262k, long400: prompt 1526 tok/s (from 1371 before the path), decode 99.2 (run-to-run
  spread today 99-104), V100 busy 8.22 s vs 3090 6.86 s: the V100 is still the slowest stage. Reply opens the same.
- Split 10,29 with the f16 path: 1538 / 98.8, V100 8.07 s: no better than 10,28.
- **Done:** in place (v70_inplace: every routed expert resident, gate|up contiguous): no group gather, a per-expert
  pointer table, one gate|up and one down launch per layer. Prompt 1526 -> 1581 tok/s, V100 busy 8.22 -> 7.78 s,
  reply bitwise the same as the per-group launches (md5). Profile before it (per group): x.gu 1337, x.down 866,
  x.gather 1113 ms (60 layers, sync-inflated).
- **Done:** DeltaNet prompt recurrence in FP32 SIMT on cc 7.0 (src/prefill/gdn70.cu, from `<scratchpad>/gt/gdn70/`,
  modeled on sFNX gdn_prompt.cuh; gdn_recurrence dispatches to it on sm70_current()). Standalone at 8192 tokens:
  22.47 -> 6.31 ms per layer chunk, error vs a double reference 3.8e-3 -> 5e-7 (q/k stay FP32). End to end
  (long400): prompt 1581 -> 1672 tok/s, decode 100.8, V100 busy 7.78 -> 7.22 s.
- 199,003-token prompt (`<scratchpad>/huge190.json`), same build: prompt 2085 tok/s, decode 92.6. Stages equally
  busy (88-92 s of 94.6 s); CUDA0/CUDA1 stream ~42k/40k experts; hand-off CUDA0 18.7 s, CUDA1 37.2 s. Target: the
  user's headless Linux two cards, ~4,000 tok/s on a ~190k prompt (decode >100, from memory).
- `--prefill 16384` on three cards: every stage borrows 4.57 GiB of cache slots, and the V100's init then falls back to
  6144-token chunks for ALL stages (one o.prefill_chunk, prompt.cpp ~433-500). Per-card chunks (16k on the 30-series,
  8k on the V100) would need the V100 stage to split each incoming chunk; parked: on the 199k prompt the V100 is
  91.8 s busy of 94.6 s (its own compute, ~0.46 ms/token), a ~2.2k tok/s ceiling that bigger chunks elsewhere don't lift.
- Repacked expert layouts (+20% standalone) are parked: the V100's expert cache is also what decode reads (ggml layout).
- **Done:** V100 dense GEMM rewritten (src/prefill/gemm_sm70.cuh replaces gemm_wmma.cuh; from `<scratchpad>/gt/gemm70/`):
  mma.m8n8k4, two smem stages + register prefetch, one barrier per step, padded/interleaved smem, grouped tile
  order, float4 epilogue; tiles by N (256x128, 128x64, 64x48, CUDA cores for N <= 8). Standalone at 8192 rows:
  gdn.qkv 11.4 -> 5.9 ms, qsa.q 14.1 -> 7.2, gdn.out 4.6 -> 3.3, hc.down 1.8 -> 1.1, hc.up 1.6 -> 1.2; bitwise the
  old kernel on every tensor-core shape; inject (N=4) now exact in FP32 (numerics change). End to end: 18k prompt
  1672 -> 1733 tok/s, decode 105.7, V100 busy 7.22 -> 6.76 s; 199k prompt 2085 -> 2213 tok/s, decode 94.4.
- **Done:** QSA prompt attention for K8V4 on cc 7.0 (src/kernels/cuda/qsa_prompt_attn_v70k8v4.cuh, from
  `<scratchpad>/gt/attn70/`; launch70<3> picks it). The old kernel was latency-bound (55.8 KB smem, 1 block/SM;
  skipping its mma saved nothing). The new one: V raw q4_0 in smem, K fragments from the pool one chunk ahead, no
  zero rows, exact 4-bit codes split hi+lo, 22.8 KB smem, 3 blocks/SM. Standalone 105.5 -> 24.8 ms per 8192-token
  chunk per QSA layer; error vs fp64 1.6e-4 -> 3.7e-7 (numerics change). Multi-query union blocks: not worth it (L2
  absorbs the shared gathers). End to end: 18k 1733 -> 1898 tok/s, decode 102.8, V100 busy 6.76 -> 5.93 s; 199k
  2213 -> 2499 tok/s, decode 93.9.
- **Done:** `--debug-layers N` (v100_howto.md): 12 layers load in ~47 s instead of ~2 min 15 s.
- **Done:** sm_70 cubin hot reload (BASALT_SM70_HOT=1): the engine reloads changed cubins between requests, so a
  V100 kernel change needs `--target sm70_cubins` and the next request, not a restart (see v100_howto.md).
- Two cards (5070 Ti + 3090, 262k): 1301 / 68.0 on Windows (bin\basalt.exe, earlier). Same build as three cards
  (serve2_iq3_v100build.bat, auto split 0-22 / 23-47, 44% of experts in RAM), long400: 1271 / 70.9. Three cards:
  1581 / ~100. Three cards only run on Windows (driver); the user's headless Linux two-card is faster than Windows
  two-card (numbers not shared).
Specs (V100 PCIe / 3090): FP32 14 / 36 TFLOPS; FP16 tensor fp32-acc ~112 / ~71; INT8 dp4a ~56 / int8 TC ~284;
bandwidth 900 / 936 GB/s. The V100 wins only on fp16 tensor cores.
Ranked queue (agent estimates, unmeasured unless noted):
Decode (per window; V100 stage 12.8 ms of 22.9 verify):
1. Dense GEMV for sm_70 in sFNX's q8s shape (warp per 1-4 rows, 16 B loads, T-templated, q|k|v segments in one
   launch): dense_tc.cu is excluded from sm70 cubins, so V100 runs native_mmvq exact (one row per 128-thread CTA)
   for qkv 1.18 + QSA q 0.45 + out-proj 0.68 ms. ~0.8-1.1 ms.
2. Shared expert folded into the glue gate/up and down launches (~6 launches/layer): ~0.5 ms.
3. Head proxy scoring dp4a-native on sm_70 (q6k_head.cu emulates mma_s8_16832 with shuffles): ~0.25-0.4 ms.
4. Router GEMV + top-10 + route in one launch (last-block pattern): ~0.15-0.2 ms.
5. One GDN rec launch on sm_70 (atomic per-head counter for the norm, z merged into qkv): ~0.15-0.25 ms.
6. QSA: fewer projection launches; top-k histogram atomics aggregated with __match_any_sync: ~0.25 ms.
7. Single-launch hc read (sFNX k_hc: grid = SMs, grid barriers): up to ~1.5 ms more; high effort.
Drafts: A/B fixed T=8 vs the cum_adapt cutoff (avg T 2.9 here; stages barely depend on T): +5-15% prose, low effort.
Prompt (V100 ~9.5 s busy per 18k):
1. Expert products as fp16-dequant m8n8k4 HMMA (sFNX moe_mma.cuh: weights to fp16 in smem with the scale folded,
   fp16 activations, 64x64x64 tiles, one launch per layer over a job list, no gather): ~1.4-1.8 s. IQ3_S blocks are
   2-byte aligned: repack planes for 16 B loads. Then swiglu in the gate/up epilogue (fp16 H): ~0.7 s more.
2. GDN chunked recurrence as FP32 SIMT (sFNX gdn_prompt.cuh; no tensor cores): gdn_chunked.cu emulates bf16/tf32
   mma with shuffles on sm_70. ~0.45-0.55 s of gdn.rec 0.78 s.
3. WMMA GEMM tuning (double buffer, BK=64, skinny tiles, merged alpha|beta): ~0.35-0.5 s.
4. QSA prompt attention with single f16 q/p (drop hi/lo): up to ~0.2-0.4 s, numerics risk.
5. QSA scorer on WMMA f16 hi/lo: ~0.06-0.09 s.

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

## Split sweep at 199k (IQ3_S, 262k, `huge190`, attention build, one run each)

Each card's prompt path lends a fixed 3.02 GiB of expert cache (8192-token chunk) unless it has that much free;
the lent experts stream from RAM every chunk, whatever the layer count. Full-attention layers are l % 4 == 3.

| Split | 5070 Ti / 3090 / V100 layers (attn) | Loans | Prompt | Decode | Stage busy s (0/1/2) |
|---|---|---|---|---|---|
| 10,28 | 10 (2) / 18 (5) / 20 (5) | 5070 Ti, 3090 | 2499 | 93.9 | 72.9 / 76.0 / 76.1 |
| 11,29 | 11 (2) / 18 (5) / 19 (5) | 5070 Ti, 3090 | 2528 | 88.1 | 72.0 / 74.5 / 74.8 |
| 12,28 | 12 (3) / 16 (4) / 20 (5) | 5070 Ti only | 2419 | 84.2 | 74.9 / 78.2 / 78.6 |
| 9,28 | 9 (2) / 19 (5) / 20 (5) | 5070 Ti, 3090 | 2283 | 91.9 | 78.3 / 82.2 / 83.1 |
| 10,26 | 10 (2) / 16 (4) / 22 (6) | 5070 Ti only | 1934 | 85.8 | 92.5 / 98.6 / 99.5 |

- Stage busy includes waiting (an upstream stage waits for its successor's previous chunk; the V100's time moved
  with upstream splits on the same 20 layers), so busy does not name the bottleneck.
- Moving the 3090's loan to the 5070 Ti (12,28) or ending it via the V100 (10,26) does not help: the streamed
  experts are not what bounds the 3090. 10,28 and 11,29 stay best; the next gains are per-layer speed on all
  three cards, the attention layers at long context first.
- `serve3_split.bat` uses `%~1`: cmd splits an unquoted `10,28` at the comma. Per-split launchers `serve3_sXXYY.bat`.

## Q5_0 and Q2_0 down on the V100 (262k, 18k prompt `long400`, one run each)

- The fp16 expert path took IQ4_NL down only. IQ4_XS's V100 layers 31, 33, 36-41, 45-47 have Q5_0 down, which MMQ
  lacks, so every card ran them on the FP16 dequant path (~20 of 80 SMs on the V100). IQ3_S layers 30, 36 (Q2_0
  down) ran MMQ on the V100.
- **Done:** TileQ5_0 and TileQ2_0 in src/prefill/moe_v70.cu; `down()` takes the type. mmq_plan marks a layer
  `2` (MMQ on cc 7.0 only) when its gate|up fits MMQ, its down does not, and moe_v70 takes both; other cards keep
  the FP16 path (`mmq_layer(l)`). Startup line: "prefill dequant GEMM (cc 7.0: f16 tiles)".
- Standalone (`<scratchpad>/gt/v70down/`, 512 experts x 2560 x 640, 81,920 rows, V100): IQ4_NL 7.49 ms, Q5_0
  8.02 ms, Q2_0 6.53 ms; vs a double reference rel rms 2.1e-4 / 2.0e-4 / 5e-7.
- `--debug-layers 8 --layer-split 2,3` IQ4_XS (V100 layers 3-7, four Q5_0): V100 expert parts 7.2 s -> 0.5 s
  per profile window; prompt 1487 -> 3283 tok/s (profiled).
- Full model: IQ4_XS 10,30: 1078 / 67.0 -> **1418 / 66.7**, V100 busy 10.9 -> 5.3 s; the 3090 (8.2 s, 8960
  streamed) is now the slowest stage. IQ3_S 10,28: 1898 / 101.6 (V100 busy 5.93 -> 5.68 s). Replies read normally.
  Numerics changed on those layers (f16 weights); the PPL/KLD check is still owed.
- Next for IQ4_XS: the V100 has 2.46 GiB spare beyond its buffers; 10,29 would move a layer off the 3090.

## IQ4_XS 10,29 and a V100 lending bug (262k, 18k prompt `long400`, one run each)
| Split | Prompt tok/s | Decode | V100 busy | V100 streamed | Reply |
|---|---|---|---|---|---|
| 10,30 | 1,420 | 67.5 | 5.3 s | 0 (no loan) | fine |
| 10,29 before fix | 493 | 112 | 22.4 s | 2,682 | "!!!!" |
| 10,29 fixed | 438 | 57.9 | 27.7 s | 3,599 | fine |
- With 19 layers the V100 fills its free memory with cache and its prompt path borrows 1,634 slots (3.78 GiB);
  those experts stream over Gen2 x1 every chunk. 10,30 is the IQ4_XS split: the V100 must not lend.
| 10,29, V100 keeps its buffers | 1,524 | 69.3 | 5.4 s | 0 | fine |
- The V100 had 4.79 GiB free for 3.78 GiB of buffers but the "keep own buffers" rule wanted 1.5 GiB more. Later
  stages now need 0.75 GiB (CUDA0, which builds the draft head after, keeps 1.5). ~0.4 GiB was left on the V100
  after a 199k prompt.
- 199k (`huge190`): 10,29 2,454 tok/s / 65.6 decode (busy 75.1 / 77.5 / 72.0 s) vs 10,30 2,233 / 64.0
  (82.5 / 85.1 / 71.7 s). 10,28 would make the V100 lend again (1.47 GB more experts).
- Bug: the cc 7.0 in-place path checked `cache->slot_of` (the cache's own table), so lent slots, holding the
  prompt buffers, were read as weights. Now it checks this read's residency (`host_res`). No change when nothing
  is lent.

## Q5_0 on MMQ (IQ4_XS 10,29, 262k, one run each)
- MMQ lacked Q5_0, so the 5070 Ti and 3090 ran IQ4_XS's Q5_0-down layers on the FP16 dequant path. Added
  `ggml_cuda_mmq_load_tiles_q5_0` (the Q5_1 unpack minus 16, Q8_0 tile layout, D4 activations) and its instance;
  the cc 7.0-only plan value is gone (the V100 still takes moe_v70 for these layers).
- prefill_mmq_kq_test, IQ3_XXS / IQ4_XS gate|up with Q5_0 down: MMQ vs double reference rel RMS 1.0e-2 / 9.8e-3
  (plain), 1.4e-2 (outliers), same as the other formats. Bench, 16 experts: 1.5-3.6x the FP16 path.
- 18k: 1,524 / 69.3 -> **1,688 / 73.4**. 199k: 2,454 / 65.6 -> **2,571 / 64.7** (busy 71.5 / 74.1 / 72.1 s).
- Open: in that test the FP16 path is ~20% off the reference for IQ3_XXS gate|up or Q5_0 down (Q4_K / Q8_0 is
  fine) though its dequantized weights match ggml's; not found yet. No pack layer takes that path for them now.

## Shallower ring and Parsec (IQ4_XS 10,29, 18k, `--vram-reserve-mib 1536`, one run each)
- Parsec streaming needs VRAM on the 5070 Ti and 3090. With the default 256 MiB reserve (339 MiB left free) every
  stage stalls, at ring 256 and 128 alike: the earlier ring-128 "crawl" was this, not the ring. Bench with 1536.
- Ring 256: **1,747 / 90.1** (9.74 s; streamed 5,547 / 8,277). Ring 128: 1,722 / 90.5 (9.92 s; 5,065 / 7,815).
  ~7% fewer experts streamed, but less copy overlap: no gain. Ring stays 256; the switch was not kept.
- No Parsec stream, default reserve, ring 256: **1,854 / 92.6** (9.24 s; streamed 4,014 / 6,693).

## Decode at 199k without Parsec (IQ4_XS 10,29, 262k, `--profile`, one run each)
- 18k (`long400`): 84.2 tok/s, verify 23.81 ms/window (2.23 tokens). 199k (`huge190`): **80.0** (was 65.6 with
  Parsec streaming), verify 25.03. Expert hits 99.3-99.8%, CPU experts ~0: RAM is not the cost.
- Stages ms/window 18k -> 199k: 5070 Ti 3.99 -> 4.13, 3090 7.29 -> 7.79, V100 11.69 -> 12.46; GPU-reach wait 2.80 -> 3.56.
  Only QSA scores+topk grows with context (0.03/0.18/0.28 -> 0.13/0.76/1.08). The V100 is half of verify.

- 5070 Ti under WDDM (probe: 256 MiB chunks until cudaMalloc fails, then read each): CUDA reports 14,921 MiB free;
  15,360 MiB read at ~842 GB/s, the rest at 6.7 GB/s. WDDM never refuses: past VRAM it backs allocations with system
  RAM (63 GiB allocated). ~440 MiB past the reported free is real VRAM; beyond it spills (fits the Parsec stalls).

- Elastic KV upper bound (64k context: 3090 cache 8503 -> 8901 slots, RAM 9.8 -> 8.5 GiB; two runs each): 18k prompt
  1,944 -> 2,007 tok/s, decode and hits within noise. <= ~3%: not worth building for the layer split.

## The bar on this PC (user's runs)
- Basalt on a single GPU here: ~2,000 tok/s prompt, ~70 decode (user's numbers). Three cards: IQ3_S 1,898 / 101.6
  (18k), 2,499 / 93.9 (199k); IQ4_XS 1,688 / 73.4, 2,571 / 64.7. The extra cards buy decode, not prompt speed.
- This PC: Ryzen 5600X (no AVX-512), 64 GB, 5070 Ti and 3090 at PCIe 3.0 x8, V100 at 2.0 x1. The Lamina/Strata trio
  read a 16k prompt at ~990 here. Split and margin tuning is fitted to these links; a planned Threadripper changes them.

## Ideas queue
1. (done) qsa split; attention ported.
1b. (done) rebalanced: 8,28 best for prompts. Retry 8,27 / 9,28 once things shift again.
1c. User idea: read prompts on the two fast cards only (balanced cut, as the two-card run that got 1301), then hand
    the V100 layers' state (KV ~70 MB + GDN state for 18k) to the V100 for three-card decode. --prompt-dup did
    this unbalanced (5070 Ti 8 layers, 3090 40) and got 707. Make the dup prompt cut balanced and compare with 1231.
    Parked: the serve prompt split (prompt_split.cpp) is for one decode card plus helpers, not layer-split decode,
    so this needs new plumbing; its ceiling is about the two-card rate (~1300), since every chunk streams the V100
    layers' experts (~16 GB IQ3_S) from RAM into the 3090 over Gen3 x8. The three-card pipeline streams none of it.
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
- gdn: Strata's V100 path (src/prefill/gdn_chunk.cu: cumsum/kkt/wmma kernels) has a different structure from
  Basalt's gdn_chunked.cu; a port is a rewrite for <= ~0.5 s. Strata also has src/split/src/moe_mma.cuh
  (m8n8k4 expert products, 64-row jobs) - a reference if the expert tile loader gets reworked.
3. gdn_chunked on WMMA/FP16.
4. (done) drafter-row report made async.
5. Rebalance layer split once the V100 is faster (more layers to it). IQ4_XS: only with a cap on the V100 cache
   so its prompt path never borrows (10,29 lends and streams over x1).
6. Decode kernels on the V100 (qsa_dec_attn, head).

## Not to do
- Don't push anywhere. Quality check (PPL/KLD) before keeping numerics changes as default.
