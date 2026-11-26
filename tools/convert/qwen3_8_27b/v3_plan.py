"""The v3 binding plan for the Qwen3.8-27B target.

The engine's v3 adapter (``src/artifact/reader.cpp``) resolves every logical
parameter through the artifact's ``bindings`` map and reads the activation
input divisor of a projection from the ``uses`` array. This module spells out
that mapping for the 27B target:

* a plain object binds its own name;
* a fused object binds several logical names to row ranges of one physical
  object -- the attention projection's query/key/gate/value, the GDN
  projection's query/key/value/z, the GDN gate pair's a/b, the MLP's
  gate/up and the vision tower's qkv triples;
* every logical parameter of an NVFP4 projection that carries an input
  divisor appears in ``uses`` with that divisor's object.

The row ranges must tile their physical object exactly: the engine sorts the
ranges of one fused object and requires them to cover it from zero.
"""

from __future__ import annotations

from typing import Sequence

from tools.artifact.container import ObjectSpec, TensorSpec
from tools.artifact.container_v3 import V3Binding, V3Plan, V3Use

from . import inventory_nvfp4 as inventory

FULL_ATTENTION_LAYERS = inventory.FULL_ATTENTION_LAYERS
VISION_LAYERS = 27
MTP_LAYER = "mtp/layers/0/"
MMFLASH_LAYERS = 5

# The geometry of the first-generation DFlash draft, as its checkpoint states
# it.  The engine validates the package against this before it binds it.
MMFLASH_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "block_size": 16,
    "dtype": "bfloat16",
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 17408,
    "layer_types": ["full_attention"] * MMFLASH_LAYERS,
    "max_position_embeddings": 262144,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": MMFLASH_LAYERS,
    "num_key_value_heads": 8,
    "num_target_layers": 64,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 248320,
    "dflash_config": {
        "block_size": 16,
        "mask_token_id": 248070,
        "target_layer_ids": [1, 16, 31, 46, 61],
    },
}

#: The frontend resources the engine reads through the text component.
TEXT_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
#: The frontend resources the engine reads through the vision component.
VISION_RESOURCES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def _elements(shape: Sequence[int]) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def build_plan(specs: Sequence[ObjectSpec]) -> V3Plan:
    """Build the v3 bindings, uses and components for one artifact inventory."""

    shapes: dict[str, tuple[int, ...]] = {}
    names: set[str] = set()
    for spec in specs:
        names.add(spec.name)
        if isinstance(spec, TensorSpec):
            shapes[spec.name] = tuple(spec.shape)

    bindings: list[V3Binding] = []
    uses: list[V3Use] = []

    def bind(logical: str, physical: str, rows: tuple[int, int] | None = None) -> None:
        shape = shapes[physical]
        width = shape[-1] if shape else 1
        total = _elements(shape)
        if rows is None:
            begin, end = 0, total
        else:
            begin, end = rows[0] * width, rows[1] * width
        bindings.append(V3Binding(logical, physical, begin, end, total))

    def use(parameter: str, divisor: str) -> None:
        uses.append(V3Use(parameter, divisor))

    # Vocabulary endpoints and the proposal head.
    bind("text/token_embedding", "text/token_embedding")
    bind("text/output_head", "text/output_head")
    bind("text/final_norm", "text/final_norm")
    bind("proposal/head", "text/draft_head")
    bind("proposal/token_ids", "text/draft_head_token_ids")

    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        bind(prefix + "input_norm", prefix + "input_norm")
        if layer in FULL_ATTENTION_LAYERS:
            projection = prefix + "attention/query_key_gate_value"
            bind(prefix + "attention/query", projection, (0, 6144))
            bind(prefix + "attention/key", projection, (6144, 7168))
            bind(prefix + "attention/gate", projection, (7168, 13312))
            bind(prefix + "attention/value", projection, (13312, 14336))
            bind(prefix + "attention/query_norm", prefix + "attention/query_norm")
            bind(prefix + "attention/key_norm", prefix + "attention/key_norm")
            bind(prefix + "attention/output", prefix + "attention/output")
            divisor = prefix + "attention/input_projection/input_scale_divisor"
            for role in ("query", "key", "gate", "value"):
                use(prefix + f"attention/{role}", divisor)
            use(
                prefix + "attention/output",
                prefix + "attention/output_projection/input_scale_divisor",
            )
        else:
            bind(prefix + "gdn/a_log", prefix + "gdn/a_log")
            bind(prefix + "gdn/dt_bias", prefix + "gdn/dt_bias")
            bind(prefix + "gdn/convolution", prefix + "gdn/convolution")
            bind(prefix + "gdn/a_projection", prefix + "gdn/a_b_projection", (0, 48))
            bind(prefix + "gdn/b_projection", prefix + "gdn/a_b_projection", (48, 96))
            projection = prefix + "gdn/query_key_value_z"
            bind(prefix + "gdn/query", projection, (0, 2048))
            bind(prefix + "gdn/key", projection, (2048, 4096))
            bind(prefix + "gdn/value", projection, (4096, 10240))
            bind(prefix + "gdn/z", projection, (10240, 16384))
            bind(prefix + "gdn/norm", prefix + "gdn/norm")
            bind(prefix + "gdn/output", prefix + "gdn/output")
            divisor = prefix + "gdn/input_projection/input_scale_divisor"
            for role in ("query", "key", "value", "z"):
                use(prefix + f"gdn/{role}", divisor)
            use(
                prefix + "gdn/output",
                prefix + "gdn/output_projection/input_scale_divisor",
            )
        bind(prefix + "post_attention_norm", prefix + "post_attention_norm")
        bind(prefix + "mlp/gate", prefix + "mlp/gate_up", (0, 17408))
        bind(prefix + "mlp/up", prefix + "mlp/gate_up", (17408, 34816))
        bind(prefix + "mlp/down", prefix + "mlp/down")
        divisor = prefix + "mlp/gate_up_projection/input_scale_divisor"
        use(prefix + "mlp/gate", divisor)
        use(prefix + "mlp/up", divisor)
        use(
            prefix + "mlp/down",
            prefix + "mlp/down_projection/input_scale_divisor",
        )

    # The MTP draft layer: the engine names its logical parameters after the
    # checkpoint's own module path (mtp/layers/0/...).
    for role in ("input_projection", "embedding_norm", "hidden_norm", "final_norm"):
        bind(f"mtp/{role}", f"mtp/{role}")
    bind(MTP_LAYER + "input_norm", "mtp/layer/input_norm")
    bind(MTP_LAYER + "post_attention_norm", "mtp/layer/post_attention_norm")
    projection = "mtp/layer/attention/query_key_gate_value"
    bind(MTP_LAYER + "attention/query", projection, (0, 6144))
    bind(MTP_LAYER + "attention/key", projection, (6144, 7168))
    bind(MTP_LAYER + "attention/gate", projection, (7168, 13312))
    bind(MTP_LAYER + "attention/value", projection, (13312, 14336))
    for role in ("query_norm", "key_norm", "output"):
        bind(MTP_LAYER + f"attention/{role}", f"mtp/layer/attention/{role}")
    bind(MTP_LAYER + "mlp/gate", "mtp/layer/mlp/gate_up", (0, 17408))
    bind(MTP_LAYER + "mlp/up", "mtp/layer/mlp/gate_up", (17408, 34816))
    bind(MTP_LAYER + "mlp/down", "mtp/layer/mlp/down")

    # The vision tower.
    bind("vision/patch_embedding", "vision/patch_embedding")
    bind("vision/patch_embedding_bias", "vision/patch_embedding_bias")
    bind("vision/position_embedding", "vision/position_embedding")
    for layer in range(VISION_LAYERS):
        prefix = f"vision/layers/{layer}/"
        projection = prefix + "attention/qkv"
        bind(prefix + "attention/query", projection, (0, 1152))
        bind(prefix + "attention/key", projection, (1152, 2304))
        bind(prefix + "attention/value", projection, (2304, 3456))
        bias = prefix + "attention/qkv_bias"
        bind(prefix + "attention/query_bias", bias, (0, 1152))
        bind(prefix + "attention/key_bias", bias, (1152, 2304))
        bind(prefix + "attention/value_bias", bias, (2304, 3456))
        for role in ("output", "output_bias"):
            bind(prefix + f"attention/{role}", prefix + f"attention/{role}")
        for role in ("fc1", "fc1_bias", "fc2", "fc2_bias"):
            bind(prefix + f"mlp/{role}", prefix + f"mlp/{role}")
        for norm in ("norm1", "norm2"):
            for part in ("weight", "bias"):
                bind(prefix + f"{norm}_{part}", prefix + f"{norm}/{part}")
    for role in ("fc1", "fc1_bias", "fc2", "fc2_bias"):
        bind(f"vision/merger/{role}", f"vision/merger/{role}")
    for part in ("weight", "bias"):
        bind(f"vision/merger/norm_{part}", f"vision/merger/norm/{part}")

    # The first-generation DFlash draft (MMFlash).  The engine reads the
    # draft's target-layer taps and block width from the component's
    # configuration, so the package states them alongside its parameters.
    for role in ("feature_projection", "context_norm", "final_norm"):
        bind(f"dflash/{role}", f"dflash/{role}")
    for layer in range(MMFLASH_LAYERS):
        prefix = f"dflash/layers/{layer}/"
        projection = prefix + "attention/query_key_value"
        bind(prefix + "attention/query", projection, (0, 4096))
        bind(prefix + "attention/key", projection, (4096, 5120))
        bind(prefix + "attention/value", projection, (5120, 6144))
        bind(prefix + "attention/query_norm", prefix + "attention/query_norm")
        bind(prefix + "attention/key_norm", prefix + "attention/key_norm")
        bind(prefix + "attention/output", prefix + "attention/output")
        bind(prefix + "input_norm", prefix + "input_norm")
        bind(prefix + "post_attention_norm", prefix + "post_attention_norm")
        gate_up = prefix + "mlp/gate_up"
        bind(prefix + "mlp/gate", gate_up, (0, 17408))
        bind(prefix + "mlp/up", gate_up, (17408, 34816))
        bind(prefix + "mlp/down", prefix + "mlp/down")

    components = {
        "text": {
            "resources": {
                role: f"frontend/{role}" for role in TEXT_RESOURCES
            }
        },
        "vision": {
            "resources": {
                role: f"frontend/{role}" for role in VISION_RESOURCES
            }
        },
    }
    component_configs = {
        "mmflash": {
            "target": "text",
            "config": MMFLASH_CONFIG,
        }
    }
    return V3Plan(
        tuple(bindings), tuple(uses), components, component_configs
    )


__all__ = [
    "MMFLASH_CONFIG",
    "MMFLASH_LAYERS",
    "MTP_LAYER",
    "TEXT_RESOURCES",
    "VISION_LAYERS",
    "VISION_RESOURCES",
    "build_plan",
]
