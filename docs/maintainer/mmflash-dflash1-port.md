# MMFlash (DFlash first generation) port plan

Branch: `feat/mmflash-dflash1`, base `8524f67` (upstream master merge, no kvmem,
no Inkling-2-Mini work, no unresolved conflict markers).

## What MMFlash is

`FSCCS/Qwen3.8-27B-MMFlash` is a *draft* model for speculative decoding of
`Qwen/Qwen3.8-27B` on image and text inputs. It is not a standalone model.
Reference: arXiv 2611.18482, code `fscdc/MMFlash`.

| | |
| --- | --- |
| Parameters | 1.73 B (BF16), 3,460,432,504 bytes in `model.safetensors` |
| Architecture | `DFlashDraftModel` (DFlash *first* generation, block diffusion) |
| Block size | 16 |
| Target hidden states | target layers 1, 16, 31, 46, 61 (of 64) |
| Draft head | none of its own: it reuses the **target model's `output_head`** |
| sha256 | `4f04793676efd7624465101a4498244fa2eac7563017f35cbff3dd451dd0a78d` (matches upstream LFS) |

The upstream model card documents only SGLang and vLLM usage, both on a
different stack. This port targets the NInfer TP2 SM70 runtime.

## Draft forward (verified against the reference implementation)

The reference is `z-lab/dflash` `dflash/model.py`, class `DFlashDraftModel`.
Reading it changed the design in two ways that the model card does not state.

### 1. The draft attends over context *and* noise, concatenated

```
target_hidden = concat(hidden_states[1], hidden_states[16], hidden_states[31],
                        hidden_states[46], hidden_states[61])      # [5*5120]
ctx           = hidden_norm(fc(target_hidden))                     # fc: [5120, 25600]
noise         = token_embedding(mask_token_id)                    # raw input embedding
for layer in 5 draft layers:
    h = h + attn(norm(h), ctx)      # k = concat([k_proj(ctx), k_proj(h)])
    h = h + mlp(norm(h))            # v = concat([v_proj(ctx), v_proj(h)])
logits        = target_output_head(h)
draft_tokens  = argmax(logits)
```

`Qwen3DFlashAttention.forward` computes `k_ctx = k_proj(target_hidden)` and
`k_noise = k_proj(hidden_states)` and concatenates them along the sequence
axis before the attention. The same holds for V. So the draft sees the
target's hidden state at the block position *as additional context keys and
values*, not as a query input.

This is the mechanism the existing NInfer DFlash2 runtime already has:
`append_context_impl` projects the captured target features through
`feature_projection` and `context_norm`, derives the context K/V with the
draft's own `weight.key` / `weight.value`, and appends them to the draft's
context KV cache. The MMFlash port reuses that path.

### 2. The draft attention is bidirectional, not causal

`Qwen3DFlashAttention.__init__` sets
`self.is_causal = layer_type == "sliding_attention"` when `config.is_causal`
is absent. MMFlash's `layer_types` are all `full_attention`, so
`is_causal` is False and the 16 block positions attend to each other freely.

## Difference from the DFlash2 runtime that already exists

| | DFlash2 (existing) | MMFlash (this port) |
| --- | --- | --- |
| Target feature layers | 5, 19, 33, 47, 61 | **1, 16, 31, 46, 61** |
| Dynamic grouped convolution | 2 per layer | **none** (`attention_conv`/`mlp_conv` are None) |
| Candidate selector + codebooks | present | **none** |
| Token selection | codebook lookup | **argmax over the target `output_head`** |
| Draft attention | sliding window 2048 | **full, bidirectional** |

Shared: the feature-capture sink, the context append, the verify/rollback
loop, the speculative round bookkeeping, the artifact layout, the packed
query/key/value and gate/up rows.

## Design decisions

1. **Runtime-selected feature layers.** `DFlashFeatureSink::layers` is already
   runtime data (`capture_layer` looks the layer up with `std::find`), so the
   tap layers do not need a new compile-time target. The existing
   `DFlashConfig::target_feature_layers` stays as the DFlash2 default; MMFlash
   supplies its own list.

2. **Weight layout.** The draft projections are stored as `W8G32_F16S` (group
   size 32, one binary16 scale per group) exactly like the DFlash2 package in
   the official v3 artifact; norms stay BF16. The measured round-trip error on
   the real MMFlash weights is a relative maximum of 0.0039 (the theoretical
   bound is 1/254) and a cosine similarity of 0.99999.

   This matters because the draft is **replicated on both TP ranks** (see
   `bindings.cpp`: `dflash/` objects are not sharded) and each 16 GB V100 has
   only about 6.5 GB free after the target model is resident. BF16 would need
   3.2 GB per rank; W8 needs about 1.7 GB.

3. **Packed projections.** The runtime binds one packed weight per projection,
   so the converter concatenates the checkpoint's separate query/key/value
   matrices into one `[6144, 5120]` object and gate/up into one
   `[34816, 5120]` object, in the same row order DFlash2 uses. The artifact
   reader's `add()` requires every part of a logical object to live in one
   physical object, so the packing has to happen in the converter.

   The result is 43 runtime objects (22 norms, 21 projections) against
   DFlash2's 66: the missing ones are the two convolution objects per layer
   and the three selector objects.

4. **Block width.** `--spec dflash` already accepts `--draft-tokens` up to 15,
   i.e. a 16-wide block, so the verify/rollback path needs no change.

## Verification plan

A separate virtualenv with the official `dflash` package, `transformers`
5.15.0 and CPU torch runs the reference draft forward and produces the golden
values the port is compared against.

## Memory budget (per 16 GB V100, TP2)

Target model resident: about 6.5 GB free after load. Draft at W8: about
1.7 GB replicated per rank. That leaves roughly 4.8 GB for KV plus workspace,
so the KV capacity has to be reduced well below 262144 tokens for the first
working run.
