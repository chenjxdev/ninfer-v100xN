"""Persistent-object contract for the Inkling-2-Mini artifact.

The artifact keeps the checkpoint's native precision: BF16 for the dense
weights, F32 for the learned queries and the router bias, and NVFP4 for the
routed experts.

The routed experts need one object per expert. Inkling's NVFP4 experts carry
a per-expert multiplier (``amax / outer_scale_denominator``) on top of the
block scales, and the artifact's NVFP4 layout has a single weight divisor per
object. Splitting the experts per expert keeps that multiplier exact: each
object stores its own divisor instead of folding it into the E4M3 block
scales, which would have cost precision.

The four MXFP8 expert layers (20-23) keep their original FP8 codes in the
existing row-scaled-FP8 container layout. A companion BF16 object per expert
matrix stores the exact power-of-two scale for every 32-value group, avoiding
both dequantization to BF16 and any requantization error.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FP32,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    TensorSpec,
)

# The frontend resources, and the checkpoint file each one comes from.
RESOURCE_SOURCES = {
    "frontend/tokenizer.json": "tokenizer.json",
    "frontend/tokenizer_config.json": "tokenizer_config.json",
    "frontend/chat_template.jinja": "chat_template.jinja",
    "frontend/generation_config.json": "generation_config.json",
    "frontend/preprocessor_config.json": "processor_config.json",
}
RESOURCE_SPECS = tuple(ResourceSpec(name) for name in RESOURCE_SOURCES)

# The artifact's format vocabulary (tools/artifact/numeric.py); the names are
# repeated here because the shared inventory module only covers the groupwise
# formats.
NVFP4 = "NVFP4"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
FP8_ROW_SCALED = "FP8_E4M3FN_ROW_BF16S"
ROW_SCALE_LAYOUT = "row-scale-v1"

MODEL_ID = "inkling2-mini"
WEIGHTS_ID = "nvfp4"
TARGET_KEY = "inkling2_mini"

# Architecture constants (config.json, inkling2_loop_text).
HIDDEN = 2304
LAYERS = 24
VOCAB = 201024
VOCAB_UNPADDED = 200058
HEADS = 24
HEADS_KV = 8
HEAD_DIM = 128
N_REL = 16
REL_EXTENT = 1024
N_EXPERT = 128
EXPERT_HIDDEN = 1152
CONV_TAPS = 4
LOOP_CORE_START = 6
LOOP_CORE_END = 18
LOOP_PASSES = 3
LOOP_GATE_RANK = 64
ATTN_RES_LAYERS = 23
# The attention-residual query is three vectors on the full-attention layers
# and one elsewhere (the shapes come straight from the checkpoint).
ATTN_RES_QUERY_WIDTH = {4: 3, 10: 3, 16: 3, 22: 3}

# Layers whose experts are MXFP8 (F8_E4M3 codes with UE8M0 block scales).
MXFP8_LAYERS = (20, 21, 22, 23)
# Layers that carry the input norms and the input-dependent convolution taps
# instead of the single attention norm.
FULL_ATTENTION_LAYERS = (5, 11, 17, 23)


@dataclass(frozen=True, slots=True)
class LayerPlan:
    index: int
    swa: bool
    loop_core: bool
    full_attention: bool
    mxfp8_experts: bool


def layer_plans() -> tuple[LayerPlan, ...]:
    plans = []
    for index in range(LAYERS):
        plans.append(
            LayerPlan(
                index=index,
                swa=index not in FULL_ATTENTION_LAYERS,
                loop_core=LOOP_CORE_START <= index < LOOP_CORE_END,
                full_attention=index in FULL_ATTENTION_LAYERS,
                mxfp8_experts=index in MXFP8_LAYERS,
            )
        )
    return tuple(plans)


def _spec(name: str, shape, fmt: str, layout: str = CONTIGUOUS_LAYOUT) -> TensorSpec:
    return TensorSpec(name, tuple(shape), fmt, layout)


def _nvfp4_spec(name: str, shape) -> TensorSpec:
    """An NVFP4 object: block-scale layout, like the Qwen3.8 mixed profile."""
    return TensorSpec(name, tuple(shape), NVFP4, BLOCK_SCALE_LAYOUT)


def text_specs() -> list[TensorSpec]:
    specs = [
        # The vocabulary endpoints stay BF16: the artifact's row-split layout
        # only accepts the grouped formats, and a BF16 vocabulary is kept as a
        # per-device replica (the Qwen3.6-27B target does the same).
        _spec("text/embed", (VOCAB, HIDDEN), BF16),
        _spec("text/embed_norm", (HIDDEN,), BF16),
        _spec("text/final_norm", (HIDDEN,), BF16),
        _spec("text/output_head", (VOCAB, HIDDEN), BF16),
    ]
    for index in range(ATTN_RES_LAYERS):
        width = ATTN_RES_QUERY_WIDTH.get(index, 1)
        specs.append(_spec("text/attn_res/attn_query/%d" % index, (HIDDEN * width,), FP32))
    for index in range(LAYERS):
        specs.append(_spec("text/attn_res/mlp_query/%d" % index, (HIDDEN,), FP32))
    specs.append(_spec("text/attn_res/output_query", (HIDDEN,), FP32))
    specs.append(_spec("text/cross_pass/query", (LAYERS, HIDDEN), FP32))
    specs.append(_spec("text/cross_pass/kv_query", (LAYERS, HIDDEN), FP32))

    for plan in layer_plans():
        prefix = "text/layers/%d" % plan.index
        specs.append(_spec(prefix + "/attn/q", (HEADS * HEAD_DIM, HIDDEN), BF16))
        specs.append(_spec(prefix + "/attn/k", (HEADS_KV * HEAD_DIM, HIDDEN), BF16))
        specs.append(_spec(prefix + "/attn/v", (HEADS_KV * HEAD_DIM, HIDDEN), BF16))
        specs.append(_spec(prefix + "/attn/o", (HIDDEN, HEADS * HEAD_DIM), BF16))
        specs.append(_spec(prefix + "/attn/rel", (N_REL * HEADS, HIDDEN), BF16))
        specs.append(_spec(prefix + "/attn/gate", (HEADS * HEAD_DIM, HIDDEN), BF16))
        specs.append(_spec(prefix + "/attn/sink", (HEADS, HIDDEN), BF16))
        specs.append(_spec(prefix + "/attn/q_clip", (HEADS, HEAD_DIM), BF16))
        specs.append(_spec(prefix + "/attn/rel_clip", (HEADS, N_REL), BF16))
        specs.append(_spec(prefix + "/attn/rel_proj", (N_REL, HEADS_KV * HEAD_DIM), BF16))
        specs.append(_spec(prefix + "/attn/sconv", (HIDDEN, CONV_TAPS), BF16))
        if plan.full_attention:
            # Full-attention layers replace the plain K/V convolutions with
            # input-dependent taps, and use separate input norms instead of a
            # single attention norm.
            specs.append(_spec(prefix + "/attn/q_input_norm", (HIDDEN,), BF16))
            specs.append(_spec(prefix + "/attn/k_input_norm", (HIDDEN,), BF16))
            specs.append(_spec(prefix + "/attn/v_input_norm", (HIDDEN,), BF16))
            specs.append(_spec(prefix + "/attn/k_sconv_static", (CONV_TAPS, HEADS_KV * HEAD_DIM), BF16))
            specs.append(_spec(prefix + "/attn/k_sconv_proj", (CONV_TAPS * HEADS_KV, HIDDEN), BF16))
            specs.append(_spec(prefix + "/attn/v_sconv_static", (CONV_TAPS, HEADS_KV * HEAD_DIM), BF16))
            specs.append(_spec(prefix + "/attn/v_sconv_proj", (CONV_TAPS * HEADS_KV, HIDDEN), BF16))
        else:
            specs.append(_spec(prefix + "/attn/k_sconv", (HEADS_KV * HEAD_DIM, CONV_TAPS), BF16))
            specs.append(_spec(prefix + "/attn/v_sconv", (HEADS_KV * HEAD_DIM, CONV_TAPS), BF16))
            specs.append(_spec(prefix + "/attn/norm", (HIDDEN,), BF16))
        specs.append(_spec(prefix + "/mlp/norm", (HIDDEN,), BF16))
        specs.append(_spec(prefix + "/mlp/sconv", (HIDDEN, CONV_TAPS), BF16))
        specs.append(_spec(prefix + "/mlp/gate", (N_EXPERT, HIDDEN), BF16))
        specs.append(_spec(prefix + "/mlp/gate_bias", (N_EXPERT,), FP32))
        if plan.loop_core:
            for site in ("attn", "k", "v", "mlp"):
                specs.append(_spec("%s/loop/%s/down" % (prefix, site), (LOOP_GATE_RANK, 2 * HIDDEN), BF16))
                specs.append(_spec("%s/loop/%s/gain" % (prefix, site), (HIDDEN, LOOP_GATE_RANK), BF16))
                specs.append(_spec("%s/loop/%s/forget" % (prefix, site), (HIDDEN, LOOP_GATE_RANK), BF16))
        for expert in range(N_EXPERT):
            base = "%s/mlp/experts/%d" % (prefix, expert)
            if plan.mxfp8_experts:
                specs.append(_spec(base + "/w13", (HIDDEN, 2 * EXPERT_HIDDEN),
                                   FP8_ROW_SCALED, ROW_SCALE_LAYOUT))
                specs.append(_spec(base + "/w2", (HIDDEN, EXPERT_HIDDEN),
                                   FP8_ROW_SCALED, ROW_SCALE_LAYOUT))
                specs.append(_spec(base + "/w13_mxscale", (HIDDEN, EXPERT_HIDDEN // 16), BF16))
                specs.append(_spec(base + "/w2_mxscale", (HIDDEN, EXPERT_HIDDEN // 32), BF16))
            else:
                specs.append(_nvfp4_spec(base + "/w13", (HIDDEN, 2 * EXPERT_HIDDEN)))
                specs.append(_nvfp4_spec(base + "/w2", (HIDDEN, EXPERT_HIDDEN)))
    return specs


def vision_specs() -> list[TensorSpec]:
    specs = [
        _spec("vision/class_embedding", (1024,), BF16),
        _spec("vision/cls_position_embedding", (1024,), BF16),
        _spec("vision/patch_embedding", (1024, 1200), BF16),
        _spec("vision/position_embedding", (1024, 1024), BF16),
        _spec("vision/ln_pre/weight", (1024,), BF16),
        _spec("vision/ln_pre/bias", (1024,), BF16),
        _spec("vision/proj_in", (4096, 4096), BF16),
        _spec("vision/proj_out", (HIDDEN, 4096), BF16),
    ]
    for index in range(23):
        prefix = "vision/layers/%d" % index
        for part in ("ln_1", "ln_2"):
            specs.append(_spec(prefix + "/%s/weight" % part, (1024,), BF16))
            specs.append(_spec(prefix + "/%s/bias" % part, (1024,), BF16))
        for part in ("ls_1", "ls_2"):
            specs.append(_spec(prefix + "/%s/gamma" % part, (1024,), BF16))
        for part in ("q_proj", "k_proj", "v_proj", "out_proj"):
            specs.append(_spec(prefix + "/%s/weight" % part, (1024, 1024), BF16))
            specs.append(_spec(prefix + "/%s/bias" % part, (1024,), BF16))
        specs.append(_spec(prefix + "/mlp/c_proj/weight", (1024, 4096), BF16))
        specs.append(_spec(prefix + "/mlp/c_proj/bias", (1024,), BF16))
        specs.append(_spec(prefix + "/mlp/c_fc/weight", (4096, 1024), BF16))
        specs.append(_spec(prefix + "/mlp/c_fc/bias", (4096,), BF16))
    return specs


def object_specs() -> list:
    return list(RESOURCE_SPECS) + text_specs() + vision_specs()


def validate_inventory() -> None:
    """Guard the plan's shape.

    The invariant that matters is coverage -- every checkpoint tensor must
    land in exactly one artifact object -- and the converter checks that
    against the real checkpoint, so this only guards the plan itself.
    """
    specs = object_specs()
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Inkling-2-Mini inventory has duplicate names")
    if not names:
        raise ValueError("Inkling-2-Mini inventory is empty")
    experts = sum(
        1 for name in names if "/mlp/experts/" in name)
    expected_experts = (LAYERS + len(MXFP8_LAYERS)) * N_EXPERT * 2
    if experts != expected_experts:
        raise ValueError(
            f"Inkling-2-Mini inventory has {experts} expert objects, "
            f"expected {expected_experts}"
        )
