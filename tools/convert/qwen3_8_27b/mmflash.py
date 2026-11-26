"""MMFlash (DFlash first generation) draft objects for the Qwen3.8-27B target.

MMFlash is the block-diffusion draft model published as
``FSCCS/Qwen3.8-27B-MMFlash``.  It is a first-generation DFlash: five standard
Qwen3 decoder blocks that read the target's hidden states at layers
``[1, 16, 31, 46, 61]`` and propose a block of 16 tokens in one pass.

Unlike the DFlash2 package the target already understands, MMFlash has no
dynamic grouped convolution and no candidate selector, and it draws its draft
tokens from the target's own output head.  Everything else -- the packed
query/key/value rows, the packed gate/up rows, the norms -- uses the same
``dflash/`` object names and the same layout as DFlash2, so the runtime binds
and executes it through the existing code path.

The projections are stored as ``W8G32_F16S`` (group size 32, one binary16
scale per group), exactly like the DFlash2 package in the official v3
artifact.  The draft is replicated on every TP rank, so the smaller format is
what keeps a 16 GB V100 able to hold the target model, the draft and a usable
KV cache at the same time.  Norms stay BF16.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from tools.artifact.container import TensorSpec
from tools.artifact.layouts import encode_direct
from tools.artifact.numeric import BF16, W8G32_F16S
from tools.convert.common.quantize import quantize_and_encode
from tools.convert.common.safetensors import ShardReader


#: The target layer taps MMFlash reads.  DFlash2 uses [5, 19, 33, 47, 61].
MMFLASH_TARGET_LAYER_IDS = (1, 16, 31, 46, 61)
MMFLASH_BLOCK_SIZE = 16
MMFLASH_MASK_TOKEN_ID = 248070

_HIDDEN = 5120
_INTERMEDIATE = 17408
_LAYERS = 5
_QUERY = 32 * 128
_KV = 8 * 128


@dataclass(frozen=True, slots=True)
class MMFlashObject:
    """One artifact object: its name, the tensor shape and the encoded payload."""

    name: str
    shape: tuple[int, ...]
    numeric_format: str
    payload: bytes


def _packed_projection_specs() -> tuple[tuple[str, tuple[int, ...], tuple[str, ...]], ...]:
    """Objects that are assembled from several checkpoint tensors.

    The runtime binds one packed weight per projection, so the checkpoint's
    separate query/key/value and gate/up matrices are concatenated here, in
    the same row order DFlash2 uses.
    """
    specs: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []
    for layer in range(_LAYERS):
        prefix = f"layers.{layer}."
        target = f"dflash/layers/{layer}/"
        specs.append(
            (
                target + "attention/query_key_value",
                (_QUERY + 2 * _KV, _HIDDEN),
                (
                    prefix + "self_attn.q_proj.weight",
                    prefix + "self_attn.k_proj.weight",
                    prefix + "self_attn.v_proj.weight",
                ),
            )
        )
        specs.append(
            (
                target + "mlp/gate_up",
                (2 * _INTERMEDIATE, _HIDDEN),
                (prefix + "mlp.gate_proj.weight", prefix + "mlp.up_proj.weight"),
            )
        )
    return tuple(specs)


def _single_tensors() -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Objects that map one-to-one onto a checkpoint tensor."""
    names: list[tuple[str, str, tuple[int, ...]]] = [
        ("dflash/feature_projection", "fc.weight", (_HIDDEN, 5 * _HIDDEN)),
        ("dflash/context_norm", "hidden_norm.weight", (_HIDDEN,)),
    ]
    for layer in range(_LAYERS):
        prefix = f"layers.{layer}."
        target = f"dflash/layers/{layer}/"
        names.extend(
            [
                (target + "input_norm", prefix + "input_layernorm.weight", (_HIDDEN,)),
                (target + "attention/query_norm", prefix + "self_attn.q_norm.weight", (128,)),
                (target + "attention/key_norm", prefix + "self_attn.k_norm.weight", (128,)),
                (
                    target + "attention/output",
                    prefix + "self_attn.o_proj.weight",
                    (_HIDDEN, _QUERY),
                ),
                (
                    target + "post_attention_norm",
                    prefix + "post_attention_layernorm.weight",
                    (_HIDDEN,),
                ),
                (
                    target + "mlp/down",
                    prefix + "mlp.down_proj.weight",
                    (_HIDDEN, _INTERMEDIATE),
                ),
            ]
        )
    names.append(("dflash/final_norm", "norm.weight", (_HIDDEN,)))
    return tuple(names)


def validate_mmflash_config(config: dict) -> None:
    """Reject any checkpoint whose geometry the runtime cannot execute."""
    draft = config.get("dflash_config", {})
    if config.get("architectures") != ["DFlashDraftModel"]:
        raise ValueError("MMFlash checkpoint must declare the DFlashDraftModel architecture")
    if config.get("hidden_size") != _HIDDEN or config.get("intermediate_size") != _INTERMEDIATE:
        raise ValueError("unsupported MMFlash hidden/intermediate width")
    if config.get("num_attention_heads") != 32 or config.get("num_key_value_heads") != 8:
        raise ValueError("unsupported MMFlash attention head layout")
    if config.get("head_dim") != 128 or config.get("num_hidden_layers") != _LAYERS:
        raise ValueError("unsupported MMFlash layer or head geometry")
    if config.get("layer_types") != ["full_attention"] * _LAYERS:
        raise ValueError("MMFlash draft layers must be full attention")
    if config.get("sliding_window") is not None or config.get("use_sliding_window"):
        raise ValueError("MMFlash draft layers must not use a sliding window")
    if tuple(draft.get("target_layer_ids", ())) != MMFLASH_TARGET_LAYER_IDS:
        raise ValueError("unsupported MMFlash target layer taps")
    if draft.get("mask_token_id") != MMFLASH_MASK_TOKEN_ID:
        raise ValueError("unsupported MMFlash mask token")
    if config.get("block_size") != MMFLASH_BLOCK_SIZE:
        raise ValueError("unsupported MMFlash block size")


def _encode_norm(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"expected a BF16 norm, got {tensor.dtype}")
    return encode_direct(tensor.to(torch.bfloat16).contiguous(), BF16)


def build_mmflash_objects(
    model: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> list[MMFlashObject]:
    """Read an MMFlash checkpoint and return its artifact objects."""
    root = Path(model)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    validate_mmflash_config(config)

    reader = ShardReader.from_file(root / "model.safetensors")
    required = {source for _, source, _ in _single_tensors()}
    for _, _, sources in _packed_projection_specs():
        required.update(sources)
    missing = sorted(required - set(reader.names))
    if missing:
        raise ValueError(f"MMFlash checkpoint is missing tensors: {missing[:4]}")

    objects: list[MMFlashObject] = []
    with reader:
        for name, source, shape in _single_tensors():
            tensor = reader.get(source)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{source}: expected {shape}, got {tuple(tensor.shape)}")
            if name.endswith(
                (
                    "/query_norm",
                    "/key_norm",
                    "/input_norm",
                    "/post_attention_norm",
                    "/context_norm",
                    "/final_norm",
                )
            ):
                objects.append(
                    MMFlashObject(name, shape, BF16, _encode_norm(tensor))
                )
                continue
            payload = quantize_and_encode(
                tensor.to(torch.float32), W8G32_F16S, device=device
            )
            objects.append(MMFlashObject(name, shape, W8G32_F16S, payload))

        for name, shape, sources in _packed_projection_specs():
            # The checkpoint stores query/key/value and gate/up separately; the
            # runtime binds one packed weight per projection, so the rows are
            # concatenated in the order DFlash2 uses.
            fused = torch.cat([reader.get(source) for source in sources], dim=0)
            if tuple(fused.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(fused.shape)}")
            payload = quantize_and_encode(
                fused.to(torch.float32), W8G32_F16S, device=device
            )
            objects.append(MMFlashObject(name, shape, W8G32_F16S, payload))
    return objects


def mmflash_specs(objects: list[MMFlashObject]) -> list[TensorSpec]:
    """Return the artifact tensor specs for the given objects."""
    return [
        TensorSpec(obj.name, obj.shape, obj.numeric_format, "row-split-k128-v1")
        for obj in objects
    ]


__all__ = [
    "MMFlashObject",
    "MMFLASH_BLOCK_SIZE",
    "MMFLASH_MASK_TOKEN_ID",
    "MMFLASH_TARGET_LAYER_IDS",
    "build_mmflash_objects",
    "mmflash_specs",
    "validate_mmflash_config",
]
