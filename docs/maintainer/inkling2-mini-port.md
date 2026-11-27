# Inkling-2-Mini → NInfer port plan

Status: TP2 text, CPU vision, official rnn_hidden MTP verification/rollback, adaptive speculation and the persistent HTTP service are working. The 120 t/s target has a verified 129.04 t/s predictable-sequence result; ordinary-chat throughput remains lower and is reported separately.
Source checkpoint: `aday777/Inkling-2-Mini-Abliterated-NVFP4` @ `40bf42fd…`
(19,525,870,752 B, 1244 tensors, single shard).

## Current implementation checkpoint (2026-11-22)

- The branch was rebuilt on `8524f67`, before the accidental re-introduction of the experimental
  KVMem line. The two Inkling converter commits and `7f49fc5`'s vision-converter fix were retained.
- `src/targets/inkling2_mini/impl/load/bindings.cpp` binds the converter's native NInfer v2
  inventory directly. It validates all 8241 objects, preserves 6144 independent expert matrices and
  1024 MXFP8 group-scale objects. No v3 reader adapter is needed.
- The execution-aware TP2 shard map and real two-V100 materialization test pass. It uploaded 6790
  text tensors and five resources; each device owned 11,442,959,360 bytes of weights and retained
  5,158,469,632 bytes free after the upload. w13 uses paired gate/up row shards, w2 uses input-column
  shards, and router/relative-bias tables remain replicated.
- The topology unit test covers 24 physical weight sets, 48 independent KV/convolution state slots,
  execution order (prelude, three core passes, coda), and native cross-pass query order
  `q[2*i], kv[2*i], kv[2*i+1], q[2*i+1]`.
- V100 numerical tests pass for `situ_mul`, `l2_clip`, `inkling_conv1d`, `inkling_depth_mix`,
  `inkling_loop_control`, `inkling_attention_dense`, `inkling_router`, `inkling_moe`, and
  `inkling_pointwise`. The dense attention test includes the SWA-1024 boundary. The MoE test covers
  both independent-divisor NVFP4 and BF16 experts and the native interleaved w13 gate/up rows.
- A 19,686,861,312-byte artifact was rebuilt and checked against the source: sampled NVFP4 maximum
  error was `2.98e-8`, sampled MXFP8 reconstruction error was zero, dense samples were byte-identical,
  and the validate-only Binder test consumed all 8241 objects. MXFP8 codes and exact power-of-two
  group scales remain packed instead of expanding to BF16. The current artifact is retained for testing;
  the source checkpoint and conversion report remain, and conversion takes about 30 seconds. Its
  preprocessor resource was replaced with valid official JSON after the original file was found to
  contain an hf-mirror 502 HTML page.

The eager runtime now implements the complete forward pass, including all 48 logical state slots,
cross-device residual blocks, three middle-cycle passes, native tokenizer encode/decode, greedy LM
head sampling, and EOS handling. Conversion recipe v3 also fixes the relative-bias transpose and
full-attention static-tap reversal. C++ prompt IDs match Hugging Face exactly for the verified chat
prompt. The verified output begins:

```text
<|message_model|><|content_thinking|>The user asks who I am. I should answer honestly and con
```

The initial correctness run took 140.6 seconds for 16 tokens (0.114 token/s). Fused top-8 MoE
scheduling, warp GEMV, K16 NVFP4 decode, bit-level FP4/FP8 conversion, parallel router/loop control,
and the TP2 pair runtime raised text decode to 51.79 token/s. The CPU vision tower recognizes the
verified red-square/blue-circle image correctly. Multi-token BF16 kernels now read each projection
weight once for all eight verify positions, direct multi-query depth mixing removes pack/extract
kernels, and a 16-entry FP4 decode table reduced eight-token FFN time from about 39 ms to 18.7 ms.
MoE route partials are reduced in fixed route order rather than by nondeterministic atomic accumulation.
Native single-token text decode is about 58.2 token/s.

The official 635,801,416-byte `mtp.safetensors` is packaged as a 54-tensor native companion
artifact. The three-layer `rnn_hidden` draft matched an independent PyTorch proposal oracle at all
seven positions. Eight-position target verification, per-token convolution history and rollback,
partial rejection, bonus-token commit, and target-feature commit are implemented. Text and vision
paths each matched their non-speculative output IDs in the 16-token parity runs. A predictable
single-request number-continuation decode reached **129.04 token/s** at gamma=5 with 24/35
accepted drafts (4.43 delivered tokens per verify), excluding prefill and without concurrency
aggregation. Ordinary open-ended chat has substantially lower acceptance and uses measured native
fallback; its speed is not represented by the 129.04 token/s result. The final persistent API
regression measured 63.16 token/s on adaptive open-ended chat, 60.08 on repeated text, and 58.63
on a 64-token vision request; a separate 37-token request reached EOS and returned `finish_reason=stop`.
Paged KV allocation, sampling beyond greedy, and token-for-token llama.cpp parity remain pending.

## Run on the two-V100 host

```bash
bash /data/models/run-inkling2-mini.sh "Hello, who are you?" 16 1024
bash /data/models/run-inkling2-mini.sh --image /data/models/strata-runs/vision-test.png \
  "What shapes and colors are in the image?" 32 256
```

The script calls `ninfer-inkling2-mini --chat` with the retained
`/data/models/inkling2_mini_nvfp4.ninfer` artifact and, by default, the
`/data/models/inkling2_mini_mtp.ninfer` companion. Set `NINFER_INKLING_MTP=` for the native
non-speculative path, `NINFER_INKLING_GAMMA=5` for the measured short-window configuration, and
`NINFER_INKLING_SPEC_ALWAYS=1` to disable adaptive native fallback. `--text` and comma-separated
token-ID modes are also available. The default is TP2 pair execution; set `NINFER_INKLING_TP2=0`
for the Pipeline16 fallback. Both use dense BF16 KV with a caller-selected context limit. Vision
runs on the CPU and injects projected embeddings into the TP2 prefix.

For the persistent combined service:

```bash
NINFER_INKLING_GAMMA=5 bash /data/models/serve-inkling2-mini.sh 8080 262144
curl -fsS http://127.0.0.1:8080/health
```

The service loads the model once and exposes `GET /v1/models`, `POST /generate`, and
`POST /v1/chat/completions` with both JSON and incremental OpenAI SSE responses; text and
base64/data-URL image requests use the same speculative
runtime. Requests are serialized because each generation resets the single-session KV/convolution
state.

Long-prompt execution uses 128-token prefill chunks. Forty sliding-window target KV slots use a
1088-entry ring instead of allocating the full context; the two local draft layers use 1024-entry
rings. Target features are committed to the draft after every block and reuse a fixed 64-token
buffer. Layers 20-23 retain exact MXFP8 codes and BF16 power-of-two group scales, saving about
1.9 GiB per GPU. Eight full-attention KV slots use INT8 codes with per-token/head FP32 scales,
which are decoded directly inside the attention kernel. The service now runs at **262144 context**,
using about 15543 MiB on GPU0 and 13543 MiB on GPU1. A real 16014-token streaming request also
completed successfully.

Prefill work awaiting the next verification window: non-final chunks skip the full vocabulary LM
head and argmax; >8-token MoE now uses device-side counting sort and four-token expert tiles,
with host-known launch bounds and no route-table D2H/H2D synchronization. Original token/route IDs
are preserved for deterministic reduction, and job/order/count metadata is preallocated in the
existing arena. `NINFER_INKLING_PREFILL_WARP_MIX=1` enables an alternative low-shared-memory warp
depth-mix path for A/B testing. These paths are implemented locally but have not been timed or
validated on the server yet.

## 1. Why this is a new target, not a new profile

Inkling-2-Mini is `model_type: inkling2_loop`, a different architecture family
from everything the engine knows:

| | Inkling-2-Mini | Qwen3.8-27B (the only tuned target) |
|---|---|---|
| hidden / layers / heads | 2304 / 24 / 24+8 | 5120 / 64 / 24+4 |
| vocab | 201024 (unpadded 200058) | 248320 |
| experts | 128 routed, 8 per token, 0 shared | dense |
| positions | **relative-position bias, no RoPE** | RoPE |
| window | SWA 1024 on a per-layer pattern | full attention |
| extra state | short conv (4 taps) on K/V/attn-out/FFN-out | gated delta net |
| repeat | **core layers 6..17 run 3 passes** with down/gain/forget controllers | none |
| extras | attention sink, QK L2 clip, relative L2 clip, log-scaled positions, attention-residual mixing, softcaps (gate 4.0 / up 25.0 / logits 30.0) | none |
| vision | `inkling2_vision`, 23 layers, hidden 1024, patch 20×20×3 | Qwen ViT |
| tokenizer | **tiktoken** (`tokenizer.model`) | Qwen BPE |
| drafter | **rnn_hidden MTP** (official `mtp.safetensors`, native runtime working) | MTP in the artifact |

The converter writes the generic native NInfer v2 container, whose reader accepts the Inkling
identity and objects without a model-specific adapter. External v3 Qwen compatibility adapters in
`src/artifact/reader.cpp` are not on this path. Inkling still needs its own
`src/targets/inkling2_mini/` binding and execution package.

## 2. Reference implementation

llama.cpp (`ggml-org/llama.cpp` @ `225a5345bbb0e0c70859af5441ba922beb666fcf`,
including the scale and attention-mask fixes) is the numerical reference:

| file | lines | content |
|---|---|---|
| `src/models/inkling2.cpp` | 460 | hparams, tensors, forward |
| `src/models/inkling2-loop.cpp` | 115 | the loop wrapper (thin) |
| `src/models/dspark.cpp` | 873 | the DSpark drafter |
| `conversion/inkling2.py` | 386 | HF → GGUF mapping (authoritative tensor map) |

The loop is `layer_phys(il) = il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start)`:
24 physical layers expand to **48 KV cache slots** (`n_slot = n_phys + (n_passes-1)*(core_end-core_start)`).

## 3. Source tensor layout (verified against the safetensors header)

Per-layer groups (24 layers, all with 128 experts):

| layers | features | experts |
|---|---|---|
| 0-4, 18, 19 | `attn_norm` | NVFP4 |
| 5 | input norms (q/k/v) + static sconv taps + `weight_proj` | NVFP4 |
| 6-10, 12-16 | **loop controllers** + `attn_norm` | NVFP4 |
| 11, 17 | loop controllers + input norms + static sconv | NVFP4 |
| 20-22 | `attn_norm` | **MXFP8** |
| 23 | input norms + static sconv | MXFP8 |

Shape facts that pin the inventory:

- `wq_du` [3072, 2304] = 24 heads × 128; `wk_dv`/`wv_dv` [1024, 2304] = 8 kv heads × 128
- `wr_du` [384, 2304] = 16 relative positions × 24 heads
- `rel_logits_proj.proj` [16, 1024]; `attn_sink_proj` [24, 2304]
- `q_l2_clip.scale` [24, 128]; `rel_l2_clip.scale` [24, 16]
- conv weights `[C, 1, 4]`; static taps `[4, C]`; tap projections `[32, 2304]`
- `w13_weight` U8 [128, 2304, 1152] → per expert a **2304×2304** NVFP4 matrix (gate‖up, 1152 each)
- `w2_weight` U8 [128, 2304, 576] → per expert a **2304×1152** NVFP4 matrix (down)
- `w13_weight_scale` F8_E4M3 [128, 2304, 144] → per expert 2304×144 = K/16 block scales
- `w2_weight_scale` F8_E4M3 [128, 2304, 72]

Non-layer tensors: `embed`/`unembed` [201024, 2304], `embed_norm`, `norm`,
`attn_res.attn_queries.{0..22}` (some [3, 2304]), `attn_res.mlp_queries.{0..23}`,
`attn_res.output_query`, `middle_cycle_cross_pass_queries` and
`middle_cycle_cross_pass_kv_queries` [24, 2304].

## 4. Quantization semantics (the part that bites)

**NVFP4 experts** (layers 0-19):

```
W = e2m1(code) * e4m3(block_scale) * (amax / outer_scale_denominator)
```

`amax` is **per expert** (F32[128]) and the denominator is per layer (F32[1]).
Measured: the per-expert multiplier spans 3.9e-5 … 2.1e-4 with **128 distinct
values** — so it cannot be folded into a single per-tensor divisor.

The engine's NVFP4 layout (`blockscale-k16-m128x4-v1`) has exactly one weight
divisor per object, but the A3B target already stores **one object per
expert** (`experts/{id}/gate` …) and aggregates them through logical aliases
(`reader.cpp:1169 add_moe`). So Inkling's experts become **6144 NVFP4 objects**
(24 layers × 128 experts × {w13, w2}), each with its own divisor
`denominator / amax` — **lossless**, no kernel change for the multipliers.

**MXFP8 experts** (layers 20-23): `w13_weight` is F8_E4M3 and the companion
`_weight_scale` is **U8 UE8M0** (2^(byte-127), 255 = NaN) with one scale per
32 elements. The engine has no UE8M0 block-scale layout, so this needs either
a new artifact format or a dequantize to BF16 (4 layers → +4 GB artifact,
+2 GB per card, which does not fit beside a 256K KV pool).

## 5. VRAM budget

Weights 19.53 GB → 9.77 GB per card at TP2. A 256K INT8 KV pool costs
6.16 GiB per card. 9.77 + 6.16 = 15.93 GiB against 16.0 GiB of usable VRAM
→ **the 256K configuration does not fit**; roughly 200K is the ceiling unless
the experts are stored smaller.

## 6. Work breakdown

| # | piece | size | notes |
|---|---|---|---|
| 1 | `tools/convert/inkling2_mini/` inventory + recipe + CLI | ~700 lines | the object layout above; experts as per-expert NVFP4 objects |
| 2 | `src/targets/inkling2_mini/` variant + bindings + package | ~1500 lines | new target; register in `registry.cpp` |
| 3 | reader admission for the new identity | ~20 lines | `reader.cpp` currently rejects anything but `qwen3.8-27b` |
| 4 | relative-position-bias attention (no RoPE) | new kernel path | the engine's attention is RoPE-only |
| 5 | SWA with a per-layer pattern | new | |
| 6 | short convolution with a recurrent state (4 taps, 4 sites) | new | state is device-resident, like the GDN state |
| 7 | MoE: 128 experts, per-expert NVFP4, gate bias, situ activation | adapt A3B | A3B has 256 experts but a different layout |
| 8 | attention sink, QK/relative L2 clip, log-scaled positions | new | |
| 9 | attention-residual mixing over depth | new | `mix()` in the reference |
| 10 | loop: 48 cache slots + down/gain/forget controllers | new | |
| 11 | softcaps (gate 4.0, up 25.0, logits 30.0) | new | |
| 12 | MXFP8 expert storage | new format or dequantize | decide with the VRAM budget |
| 13 | `inkling2_vision` tower (23 layers, hidden 1024, patch 20×20×3) | new | the Qwen ViT path does not apply |
| 14 | tiktoken tokenizer in the frontend | new | the frontend is Qwen-BPE specific |
| 15 | DSpark drafter | ~800 lines | `mtp.safetensors` is a separate checkpoint |

Items 4-12 are the bulk and each needs its own sm_70 kernel work; the
reference implementation is the guide for the semantics, not a port that
copies code.

## 7. Alternative worth weighing first

`ggml-org/Inkling-2-Mini-GGUF` already exists, and llama.cpp runs this model
today. On this box that path needs no ninfer work at all, and it also gives a
numerical oracle for the port. The ninfer path is worth it only if the
engine's V100-specific kernels (MTP-style drafting, 256K context, vision) are
required for Inkling as well.
