# Prompt path on the layer split (2026-11-10)

IQ4_XS, `--gpu 1,2,0 --layer-split 10,29`, 262k context, no profiler, no Parsec. Power limits: V100 250 W,
3090 350 W, 5070 Ti 300 W. Method: `v100_howto.md` (Prompt-speed A/B). One run per cell unless noted.

## Why short prompts were slow
`opencode` turns read 1-4k new tokens. With `--prefill 8192` (the config) such a prompt is one chunk, and the
stages only overlap across chunks, so the cards ran one after another. 4,089 tokens: 5070 Ti 649 ms, 3090 1,469,
V100 1,169; wall 3,790 ms.

## Chunk sweep (prompt tok/s as the server reports it)
| chunk | 1.4k (p1_1) | 4.1-4.6k (p4_*) | 18k (long400) |
|---|---|---|---|
| 8192 | 827 | 950-1,020 | 1,822 |
| 3072 | 850 | 1,420-1,460 | 2,458 |
| 2048 | 841 | 1,460-1,570 | 2,402 |
| 1024 | 1,012 | 1,590-1,660 | 2,134 |

Chosen: a layer split caps the chunk at prompt/4, between 512 and 3,072 (`kPipeChunkMin/Max`, state.hpp;
prompt.cpp `lend`). Result: 1.4k 1,013, 4k ~1,590, 18k 2,458, 88k (`h80`) 2,734 (all three cards ~30 s busy,
balanced). Same expert-cache slots as before. Chunking changes K/V rounding, so replies differ from 8192-chunk
replies (docs/precision/index.md, "Prompt chunk size").

## Fixed cost per request (~250 ms -> ~12 ms)
Found by timing the read loop (see the how-to). Before, for a 4k prompt:
- turn checkpoint saved synchronously on all three cards: 126 ms;
- refill of the lent cache slots after the read: 5070 Ti 241-339 slots, 73-106 ms; 3090 ~230 slots, overlapped.

Changes, in order:
1. Each stage saves its checkpoint part through its own deferred copy (`st_ckd`, conversation.cpp): 126 -> 9 ms.
2. A stage's loan is refilled when that stage has read its layers of the last chunk (`stage_reached` hook from
   `on_stage_chunk`, prompt_read.cpp; `PfPart::early`, `refill_early`). CUDA0 alone just moved the wait to the
   3090; every stage but the last brought both waits to 0.
3. The 110 ms left was `res_upload`: the V100's `res_put` (pageable copy + legacy-stream sync) waited 105 ms behind
   the deferred checkpoint copy on its x1 link. The V100 never lends, so the table now goes only to cards whose
   rows changed (`res_upload(parts)`, tiers.cpp).

After: 1.4k 1,170 tok/s, 4k ~1,800 (server ms 1,187 and 2,252 vs 1,679 and 4,214 at session start).

## Continuing a conversation
Turn 2 of `mt.sh` (2,260 reused + 55 read): 382 -> 247 ms. The deferral's spare device buffer still holds the last
checkpoint's GDN bytes; restoring that checkpoint copies them on the device (keyed by the host buffer, size and
ids). Replies identical to a host-restore run of the same history.

## Commits
7e57f780 (chunk), cab9cfee (deferred stage checkpoints), b9e82254 (early refill + masked upload), 0c451bef (device
restore). Hashes after the author rewrite.

## Tried, no gain
- Q6_K gemv with each row staged in shared memory by 16 B loads (`<sp>/gt/q6s.cu`): bitwise, slower on all three
  cards. V100: T=1 already ~600 GB/s; the T=3 cost is the per-token dot work, not loading weights.

## Queued: decode launch fusions (audit 11-10, not started)
Per-layer decode is ~16 main + ~10 side nodes (GDN), ~18 + ~15 (QSA); each main node a layer is ~36 us a window on
the V100. Bitwise candidates, best first:
1. q8_1 quantize inside `hc8_up2` (-1 main, -1 side); copy `norm_q8`'s fast-math pattern (dec_gdn_rec.cu).
2. Shared expert: `sigmoid_scale_rows` into `dm_combine4`, gate_inp as row 512 of the router gemv, gate+up as one
   mmvq launch (-3 side).
3. `swiglu_q8_1_entries` into the `native_down_multi` prologue (-1 main; profile first).
4. `top10` into the router gemv's last CTA (-1 main).
5. GDN qkv+z one launch, split+norm fused (-1 side, -1 main).
6. QSA K+V and indexer k+q merged (-2 side); scores->topk, chunk->merge (-2 main).
7. `dec_moe_route` mode 0 instead of modes 1 and 2 (-1 side).
Expected total: ~0.1 ms a window on the V100 (~0.5%).
