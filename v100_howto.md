# How the V100 work is done

The method behind `v100_ralph.md` (results and queue live there). `<sp>` is the session scratchpad:
`C:\Users\Vincent\AppData\Local\Temp\claude\1d9e6d301502be43\af306379-64f2-456b-950b-6810d7ac7fb5\scratchpad`.

## The machine and the layer split
- Cards (PCI_BUS_ID order, `nvidia-smi` index): 0 = Tesla V100-PCIE-32GB (sm_70, TCC, chipset slot Gen2 x1, ~0.4 GB/s),
  1 = RTX 5070 Ti (sm_120, 16 GB), 2 = RTX 3090 (sm_86, 24 GB). Ryzen 5600X, X370, 64 GB RAM, Windows 11.
- The server runs `--gpu 1,2,0` with `CUDA_VISIBLE_DEVICES=0,1,2`, so the engine's CUDA0 = 5070 Ti, CUDA1 = 3090,
  CUDA2 = V100 (the last stage). The drafter (MTP) is on CUDA0.
- `--layer-split K1,K2` gives the FIRST layer of each later card: `10,28` = layers 0-9 on the 5070 Ti, 10-27 on the
  3090, 28-47 on the V100. Best so far: IQ3_S `10,28`, IQ4_XS `10,30`.
- Rule for the split: the V100 must keep its own prompt buffers. engine.log line `CUDA2 keeps its own prompt buffers
  (... free): no loan`. If it borrows cache slots instead, it re-streams those experts over x1 every chunk (IQ4_XS at
  10,28 fell to 394 tok/s).
- Context is always the full 262,144 (basalt_config.json). Launchers carry no `--max-context`.
- The "IQ3_S" pack (`Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt`) is mixed-quant. Expert types on the V100's layers
  (engine.log `experts A/B, layers ...`, A = gate|up, B = down):

| layers | gate/up | down |
|---|---|---|
| 28, 29, 35, 40, 44, 45 | IQ3_S | IQ4_NL |
| 31, 33, 37, 38, 41, 42, 43, 46 | IQ3_XXS | IQ4_NL |
| 30 | IQ3_XXS | Q2_0 |
| 32, 34, 39 | IQ2_S | IQ4_NL |
| 36 | IQ2_S | Q2_0 |
| 47 | IQ4_XS | IQ4_NL |

## Building for the V100
- CUDA 13 dropped sm_70, so the engine's sm_70 code is a set of cubins built by CUDA 12.8
  (`D:\AI\cuda-12.8\root`, cmake `BASALT_SM70_NVCC`, cmake/sm70_cubins.cmake) into `build-v100/bin/sm70/`. Every .cu of
  the core, kernels, prefill and mmq targets gets one, except hc_fused.cu, dense_tc.cu and moe_fp4.cu.
- Kernels reach the V100 only through `basalt::kernels::launch_cc` (include/basalt/kernels/cc_launch.hpp): on a cc 7.0
  device it loads the kernel from the cubins by its mangled name. A plain `<<<>>>` launch would not find sm_70 code.
- `#if BASALT_SM80` gates Ampere+ instructions. Without them a kernel falls back to include/basalt/kernels/sm70_emu.cuh
  (cp.async becomes a plain copy, mma m16n8 and ldmatrix are done with shuffles, which is slow). Finding those
  fallbacks in hot loops is how the V100 targets were picked.
- Build: `cmd //c "$(cygpath -w <sp>/build-v100.bat)"`. Stop the server first: it locks basalt.exe (LNK1104).
- git needs `git -c safe.directory=D:/AI/basalt`.

## Running and measuring
- Launchers in `<sp>`:
  - `serve3_iq3_1028.bat`: plain IQ3_S, 10,28, port 8096.
  - `serve3_iq3_1028_prof60.bat`: `BASALT_PF_PROF=60`, per-part prompt profile every 60 layers.
  - `serve3_iq3_1028_dprof.bat`: `--profile`, per-stage decode GPU parts.
  - `serve3_iq4_1030.bat`: IQ4_XS.
- Start a server as a background task; stop it with TaskStop on that task. Killing basalt.exe by image name is
  blocked by the auto-mode classifier.
- The server says "ready" before the engine has loaded. Wait for `/health` to report `"loaded": true`; `measure.sh`
  does this.
- `measure.sh <tag> <server log>`: sends `<sp>/long.json` (18,096-token summary prompt, greedy, 200 out). Set
  `PROMPT=long400` for 400 out, or `PROMPT=huge` for the 125,278-token prompt. It prints prompt/decode tok/s and the reply's start.
- `th1.sh <tag>`: long400 on a running server (expects `<sp>/srv_th_<tag>.log`). It prints tok/s, the window stats
  and an md5 of the reply (greedy identity across A/Bs that should not change numerics).
- `dprof.sh <tag>`: the decode profile on a `--profile` server (expects `<sp>/srv_<tag>.log`).
- engine.log: `~/.cache/basalt/9ee2c74ced041f2c/engine.log` (IQ3_S), `.../467d531489615156/engine.log` (IQ4_XS).
  - `prompt stages:` is busy ms per card.
  - `pf-prof: CUDA2 ...` is ms per prompt part. It is inflated by the syncs, so compare relative values only.
  - `decode GPU stages, stage N` is ms per verify window per card.
  - `decode timing:` gives windows, avg T, tokens per window and verify ms.
- Nsight Systems hangs this engine on this box (through serve and through `--serve` stdin alike). Don't use it.
  Rechecked 11-10: under `nsys profile` the engine prints nothing, not even the model load, until nsys's
  `--duration` kills it. Driving `--serve` directly, send `QUIT` only after `DONE`: a line that arrives while
  the prompt is read cancels the request (`DONE ... cancel`, 0 generated).

## Prompt-speed A/B (short and long prompts)
- Prompt set in `<sp>`, each with a different prefix so no conversation-cache reuse: `p1_0`, `p1_1` (~1.4k tokens),
  `p4_0..2` (4.1k, 4.2k, 4.6k), all cut from `long400.json` with "Summarize the above in one sentence", 16 out;
  then `long400` (18,096, 400 out). `h80.json`: the first 300,000 chars of `huge.json` (88,504 tokens), 16 out.
- `ab_prompt.sh <tag> <server log> <port>`: waits for "ready", sends p1_0, p4_0, p1_1, p4_1, p4_2, long400 in that
  order, prints prompt n, prompt tok/s and server prompt ms. p1_0 is always slow (first request warms up): compare
  from p4_0 on.
- `mt.sh <tag> <port>`: a two-turn chat (`mt1.json`, then the reply plus one more user line). Turn 2 must show
  `cache_n` ~2260 (checkpoint reused). Same server history, same build: replies identical; two host-restore runs
  can still differ late in turn 2, so compare a change against more than one baseline run.
- Test servers: copy `serve4_s1029_8095.bat` with `--port 8096` (plus `--prefill N` for chunk sweeps). The user's
  server is port 8095; stop it before building (a running exe makes the link fail and leaves a stale binary).
- Per-request fixed cost: the engine's `prompt N tokens ... read in X ms` minus the same request's
  `prompt stages: ... in Y ms` is time outside the pipeline (lend, checkpoint, refill, residency upload). To split
  it, add temporary `basalt::log::info` timers around `lend`, `read_part`, `checkpoint_at`, `refill` and
  `res_upload` (request_prefill.cpp, prompt.cpp, tiers.cpp), then remove them before committing.
- Don't `git stash` here: the user's `tools/*` mode changes block the pop.

## Fast restarts: a few layers only
- `--debug-layers N` makes the engine run only the first N layers; the output is meaningless.
  - It caps the geometry and truncates the expert layout to those layers' blobs. A full profile serves the first
    layers.
  - Load takes ~47 s for 12 layers against ~2 min 15 s for 48. The head, embeddings and other dense weights still
    load.
- `<sp>/serve3_dbg12.bat`: 12 layers split `4,8`, so the V100 runs layers 8-11 (three GDN layers plus one QSA,
  the model's mix).
- Each stage carries ~2 s of fixed per-prompt cost (embedding, hand-off, per-chunk setup). Compare runs at the same
  N, or take the difference between two N, rather than scaling to 48.
- Don't run it while something else is timing kernels on the V100.

## Changing a V100 kernel without reloading the weights
- Start the server with `BASALT_SM70_HOT=1` (`<sp>/serve3_iq3_1028_hot.bat`).
- After editing a .cu file, rebuild only its cubin while the server runs. The exe is locked, but the cubins aren't:
  `cmd //c "<vcvars + CUDA_PATH as in build-v100.bat> && cmake --build build-v100 --target sm70_cubins"`.
- The next request reloads every changed cubin (engine.log `basalt sm70: CUDA2: reloaded N cubins, M kernels`).
  Prompt-path launches use the new code at once.
- Decode's captured CUDA graphs keep the old code (the old modules stay loaded), so decode changes still need a
  restart. So do host-side (.cpp) changes.
- A repeated prompt hits the conversation cache. Change the prompt, or check `cached_tokens` in the reply, when
  re-measuring prompt speed.

## Timing kernels on their own
Per-kernel truth comes from standalone programs in `<sp>/gt/`, built with CUDA 12.8 for sm_70/86/120. They run with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` (V100) or `=2` (3090).
- `bh.bat <name>` builds `<name>.cu` with nvcc 12.8 + vcvars64. The cudart and cccl includes are needed.
- Basalt's own source in a bench:
  - Copy a kernel's lines into a header (`hc8_old.cuh` from hc8_gr.cu); or
  - `#include` the .cu itself with stubs for `sm70_current()` (false) and the other cubin-loader functions, so
    launches are plain (`mmb.cu` with native_mmvq.cu; build with `bm.bat`, which adds `-I D:/AI/basalt/include`,
    C++20 and the MSVC flags).
- Time inside a CUDA graph of ~100-200 launches (best of 5), as decode runs them. Graph node gaps: V100 2.2 us,
  5070 Ti 0.8 us (`gl.cu`).
- Check every variant against the current kernel's outputs (max |diff|) before wiring anything in.
- `gt.cu`: `%globaltimer` resolution (V100 1.024 us).

## The f16 tensor-core expert kernel (prompt reading)
Written standalone first in `<sp>/gt/moe70/` (moe70.cu, harness.inc, gu_types.inc, tables.h, build.bat):
- The design follows sFNX's k_moe_mma (github.com/johnfink8/sfnx, MIT; read-only clone in `<sp>/ext/sfnx`):
  - One launch over a job list (an expert plus at most 64 consecutive sorted rows; a row's input and output share
    its index). The grid is (weight rows / 128, jobs), with 128 threads.
  - Each k-step of 64, a thread dequantizes one 32-weight sub-block of its weight row to f16, with the block scale
    folded in, straight into shared memory. int to f16 is exact via `__byte_perm(w, 0x64646464, ...)` and `hsub2`.
  - Activations are f16 rows read by index.
  - `mma.sync.m8n8k4.row.col.f32.f16.f16.f32`, f32 accumulation. Each warp owns 64 weight rows x 32 tokens.
  - The next k-step's raw weights and activations are prefetched into registers (sm_70 has no cp.async).
  - gate|up tile rows pair gate row f with up row FF + f, so the epilogue writes silu(gate) * up as f16 H. There is
    no separate swiglu or quantize kernel.
- Tile loaders (`fetch` raw bytes, `put` 32 f16 values, `load_grid` into shared memory) exist for IQ3_S, IQ3_XXS,
  IQ2_S, IQ4_XS (gate|up) and IQ4_NL (down), in ggml's stored layout. Repacked 16-byte layouts exist for IQ3_S and
  IQ4_NL (about 20% faster, not wired in yet). The IQ2_S grid sits in shared memory as nibble codes (4 KB) so
  3 blocks fit per SM.
- The harness makes random valid blocks and Zipf-skewed expert counts (81,920 rows = one 8192-token chunk x top-10).
  It checks against a CPU double-precision dequant reference (~1e-3 max relative error, ~3e-4 rms) and times every
  kernel. Run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./moe70.exe check` (or `gu` for the type sweep).
- Into Basalt:
  - `src/prefill/moe_v70.cu` holds moe70.cu's kernel side verbatim (from `constexpr int KC` to the end of
    `k_moe`). The `d_*_grid` device arrays are replaced by ggml_common.h's tables (`GGML_COMMON_IMPL_CUDA`).
  - Launchers go through launch_cc.
  - The file is listed in the prefill target in CMakeLists.txt, so its sm_70 cubin is built automatically.
- The switch is `v70_l` in src/prefill/experts.hpp: an MMQ layer on a cc 7.0 device whose types `moe_v70::usable` takes
  (gate|up IQ3_S/IQ3_XXS/IQ2_S/IQ4_XS, down IQ4_NL, no scale tail). Other layers (the two Q2_0-down layers) keep MMQ.
- Data flow in src/prefill/expert_mmq.cpp:
  - `prepare_products`: `gather_rows16(mixed_h -> GU)` puts the layer's activations as f16 rows in expert order. For
    this pack GU holds T*K*1280 floats = T*K*N halves. It then builds every group's jobs from `order`, `m.off` and
    `m.cnt` and uploads them (`moe_v70::job_buffer`).
  - `compute_mmq`, per group of MMQ_GROUP experts (already gathered into grp_gu / grp_d in ggml layout):
    `gu_swiglu` (f16 H into H, T*K*640 floats of room), then `down` into Dm. Both use absolute sorted rows, as MMQ
    did, so the rest (the router-weighted sum) is unchanged.
- Numerics change (f16 weights and activations instead of int8 dot products). The `--quality` PPL/KLD gate is still
  owed.

## The coalesced hc8 down (decode)
- hc8_gr.cu's down projection had each thread walk whole 32-column groups, so a warp's loads touched 32 cache lines.
  Lanes now take consecutive 4-column chunks over 256 threads.
- Found with `<sp>/gt/hc8b.cu` (the three hc8 kernels copied in, T = 1/3/8, V100 vs 3090). Checked with
  `build-v100/hc8_parity.exe` on the 3090 and the 5070 Ti; the test kernels have no sm_70 image.
