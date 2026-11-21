# Inkling-2-Mini → NInfer port plan

Status: research complete, engine work not started.
Source checkpoint: `aday777/Inkling-2-Mini-Abliterated-NVFP4` @ `40bf42fd…`
(19,525,870,752 B, 1244 tensors, single shard).

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
| drafter | **DSpark** (`mtp.safetensors`) | MTP in the artifact |

The reader admits `qwen3.8-27b` and `qwen3.6-35b-a3b` only
(`src/artifact/reader.cpp`), so this needs a new
`src/targets/inkling2_mini/` plus a converter inventory.

## 2. Reference implementation

llama.cpp master (`ggml-org/llama.cpp` @ `e50c254`) already implements it:

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
