"""Convert the Inkling-2-Mini NVFP4 checkpoint into a NInfer artifact.

The source is a ModelOpt-style export whose routed experts are NVFP4 with a
per-expert multiplier. The artifact stores one NVFP4 object per expert so
that multiplier stays exact: the block scales and the packed codes are copied
byte for byte and the object carries its own weight divisor.

    python3 -m tools.convert.inkling2_mini.convert \
      --model /data/models/Inkling-2-Mini-Abliterated-NVFP4 \
      --out /data/models/inkling2_mini_nvfp4.ninfer
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec as ArtifactResourceSpec,
)
from tools.artifact.layouts import encode_nvfp4
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import inventory as inv

SOURCE_PREFIX = "model.llm"
VISION_PREFIX = "model.visual"
LOOP_SITES = ("attn", "k", "v", "mlp")
DENSE_ATTN = {
    "attn/q": "attn.wq_du.weight",
    "attn/k": "attn.wk_dv.weight",
    "attn/v": "attn.wv_dv.weight",
    "attn/o": "attn.wo_ud.weight",
    "attn/rel": "attn.wr_du.weight",
    "attn/gate": "attn.gate_attn.weight",
    "attn/sink": "attn.attn_sink_proj.weight",
    "attn/q_clip": "attn.q_l2_clip.scale",
    "attn/rel_clip": "attn.rel_l2_clip.scale",
    "attn/rel_proj": "attn.rel_logits_proj.proj",
    "attn/k_sconv": "attn.k_sconv.weight",
    "attn/v_sconv": "attn.v_sconv.weight",
    "attn/sconv": "attn_sconv.weight",
}


def f32_bytes(value: float) -> bytes:
    return struct.pack("<f", value)


def source_name_for(object_name: str, plan: inv.LayerPlan | None) -> str | None:
    """Map an artifact object name back to the checkpoint tensor it carries."""
    if object_name == "text/embed":
        return SOURCE_PREFIX + ".embed.weight"
    if object_name == "text/embed_norm":
        return SOURCE_PREFIX + ".embed_norm.weight"
    if object_name == "text/final_norm":
        return SOURCE_PREFIX + ".norm.weight"
    if object_name == "text/output_head":
        return SOURCE_PREFIX + ".unembed.weight"
    if object_name == "text/attn_res/output_query":
        return SOURCE_PREFIX + ".attn_res.output_query"
    if object_name == "text/cross_pass/query":
        return SOURCE_PREFIX + ".middle_cycle_cross_pass_queries"
    if object_name == "text/cross_pass/kv_query":
        return SOURCE_PREFIX + ".middle_cycle_cross_pass_kv_queries"
    if object_name.startswith("text/attn_res/attn_query/"):
        index = int(object_name.rsplit("/", 1)[1])
        return "%s.attn_res.attn_queries.%d" % (SOURCE_PREFIX, index)
    if object_name.startswith("text/attn_res/mlp_query/"):
        index = int(object_name.rsplit("/", 1)[1])
        return "%s.attn_res.mlp_queries.%d" % (SOURCE_PREFIX, index)

    if object_name.startswith("vision/"):
        tail = object_name[len("vision/"):]
        mapping = {
            "class_embedding": "backbone.embeddings.class_embedding",
            "cls_position_embedding": "backbone.embeddings.cls_position_embedding",
            "patch_embedding": "backbone.embeddings.patch_embedding.weight",
            "position_embedding": "backbone.embeddings.position_embedding",
            "ln_pre/weight": "backbone.ln_pre.weight",
            "ln_pre/bias": "backbone.ln_pre.bias",
            "proj_in": "proj_in.weight",
            "proj_out": "proj_out.weight",
        }
        if tail in mapping:
            return VISION_PREFIX + "." + mapping[tail]
        parts = tail.split("/")
        if parts[0] == "layers":
            index = parts[1]
            rest = parts[2:]
            if rest[0] in ("ln_1", "ln_2", "ls_1", "ls_2"):
                suffix = "gamma" if rest[0].startswith("ls") else rest[1]
                return "%s.backbone.layers.%s.%s.%s" % (VISION_PREFIX, index, rest[0], suffix)
            if rest[0] in ("q_proj", "k_proj", "v_proj", "out_proj"):
                return "%s.backbone.layers.%s.self_attn.%s.%s" % (
                    VISION_PREFIX, index, rest[0], rest[1])
            if rest[0] == "mlp":
                return "%s.backbone.layers.%s.mlp.%s.%s" % (
                    VISION_PREFIX, index, rest[1], rest[2])
        return None

    if plan is None:
        return None
    prefix = "model.llm.layers.%d" % plan.index
    tail = object_name.split("/layers/%d/" % plan.index, 1)[1]
    if tail in DENSE_ATTN:
        return prefix + "." + DENSE_ATTN[tail]
    if tail == "attn/q_input_norm":
        return prefix + ".q_input_norm.weight"
    if tail == "attn/k_input_norm":
        return prefix + ".k_input_norm.weight"
    if tail == "attn/v_input_norm":
        return prefix + ".v_input_norm.weight"
    if tail == "attn/norm":
        return prefix + ".attn_norm.weight"
    if tail == "attn/k_sconv_static":
        return prefix + ".attn.k_sconv.static_w"
    if tail == "attn/v_sconv_static":
        return prefix + ".attn.v_sconv.static_w"
    if tail == "attn/k_sconv_proj":
        return prefix + ".attn.k_sconv.weight_proj"
    if tail == "attn/v_sconv_proj":
        return prefix + ".attn.v_sconv.weight_proj"
    if tail == "mlp/norm":
        return prefix + ".mlp_norm.weight"
    if tail == "mlp/sconv":
        return prefix + ".mlp_sconv.weight"
    if tail == "mlp/gate":
        return prefix + ".mlp.gate.weight"
    if tail == "mlp/gate_bias":
        return prefix + ".mlp.gate.bias"
    if tail.startswith("loop/"):
        _, site, kind = tail.split("/")
        return prefix + ".controllers.%s.%s_weight" % (site, kind)
    return None


def expert_source(plan: inv.LayerPlan, expert: int, which: str) -> str:
    return "model.llm.layers.%d.mlp.experts.%s_weight" % (plan.index, which)


def dequantize_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an MXFP8 expert: F8_E4M3 codes with UE8M0 block scales."""
    codes = weight.to(torch.float32)
    exponent = scale.to(torch.int32) - 127
    multiplier = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    shape = codes.shape
    blocks = codes.reshape(*shape[:-1], shape[-1] // 32, 32)
    scales = multiplier.reshape(*multiplier.shape, 1)
    return (blocks * scales).reshape(shape).to(torch.bfloat16)


def build_payload(object_name: str, spec, reader: ShardReader, plan, expert_cache,
                  consumed: set[str]):
    """Return the payload bytes for one artifact object."""
    if object_name.startswith("text/layers/") and "/mlp/experts/" in object_name:
        _, _, layer_text, _, _, expert_text, which = object_name.split("/")
        expert = int(expert_text)
        prefix = "model.llm.layers.%d.mlp.experts" % int(layer_text)
        for suffix in ("%s_weight" % which, "%s_weight_scale" % which,
                       "%s_weight_amax" % which,
                       "%s_weight_outer_scale_denominator" % which):
            consumed.add(prefix + "." + suffix)
        if plan.mxfp8_experts:
            weights = expert_cache.setdefault(
                (plan.index, which), reader.get(prefix + ".%s_weight" % which))
            scales = expert_cache.setdefault(
                (plan.index, which + "s"), reader.get(prefix + ".%s_weight_scale" % which))
            dequantized = dequantize_mxfp8(weights[expert], scales[expert])
            tensor = dequantized.reshape(inv.HIDDEN, 2 * inv.EXPERT_HIDDEN) if which == "w13" \
                else dequantized.reshape(inv.HIDDEN, inv.EXPERT_HIDDEN)
            return tensor.contiguous().view(torch.uint16).numpy().tobytes()
        packed = expert_cache.setdefault(
            (plan.index, which), reader.get(prefix + ".%s_weight" % which))[expert]
        scales = expert_cache.setdefault(
            (plan.index, which + "s"), reader.get(prefix + ".%s_weight_scale" % which))[expert]
        amax = expert_cache.setdefault(
            (plan.index, which + "a"), reader.get(prefix + ".%s_weight_amax" % which))[expert]
        denominator = expert_cache.setdefault(
            (plan.index, which + "d"),
            reader.get(prefix + ".%s_weight_outer_scale_denominator" % which))[0]
        divisor = float(denominator) / float(amax)
        shape = (inv.HIDDEN, 2 * inv.EXPERT_HIDDEN) if which == "w13" else (inv.HIDDEN, inv.EXPERT_HIDDEN)
        codes = packed.reshape(shape[0], shape[1] // 2)
        block_scales = scales.reshape(shape[0], shape[1] // 16)
        return encode_nvfp4(
            codes.contiguous(),
            block_scales.contiguous().view(torch.uint8),
            f32_bytes(divisor),
            shape,
        )

    name = source_name_for(object_name, plan)
    if name is None:
        raise ValueError(f"no source tensor for {object_name}")
    consumed.add(name)
    tensor = reader.get(name)
    if tensor.dtype == torch.bfloat16:
        return tensor.contiguous().view(torch.uint16).numpy().tobytes()
    if tensor.dtype == torch.float32:
        return tensor.contiguous().view(torch.int32).numpy().tobytes()
    raise ValueError(f"{name}: unsupported dtype {tensor.dtype}")


def convert(model_dir: str | Path, out_path: str | Path) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    inv.validate_inventory()
    plans = {plan.index: plan for plan in inv.layer_plans()}

    source_dir = Path(model_dir)
    resources = {
        name: (source_dir / source_file).read_bytes()
        for name, source_file in inv.RESOURCE_SOURCES.items()
    }
    specs = family_conversion.build_object_plan(
        inv.object_specs(), resources).specs

    reader = ShardReader(str(model_dir))
    consumed: set[str] = set()
    try:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inv.MODEL_ID, inv.WEIGHTS_ID),
            specs,
        ) as writer:
            expert_cache = {}
            for index, spec in enumerate(specs, start=1):
                if isinstance(spec, ArtifactResourceSpec):
                    writer.write(spec.name, resources[spec.name])
                    if index % 256 == 0 or index == len(specs):
                        print(f"[{index}/{len(specs)}] written", flush=True)
                    continue
                layer_index = None
                if spec.name.startswith("text/layers/"):
                    layer_index = int(spec.name.split("/")[2])
                plan = plans.get(layer_index)
                payload = build_payload(
                    spec.name, spec, reader, plan, expert_cache, consumed)
                writer.write(spec.name, payload)
                if index % 256 == 0 or index == len(specs):
                    print(f"[{index}/{len(specs)}] written", flush=True)
        # Every checkpoint tensor must land in exactly one artifact object.
        missing = sorted(set(reader.names) - consumed)
        if missing:
            raise ValueError(
                f"{len(missing)} checkpoint tensors were not converted, first: {missing[0]}"
            )
    finally:
        reader.close()

    elapsed = time.perf_counter() - started
    print(f"complete: {output.stat().st_size} bytes in {elapsed:.1f}s")
    report = Path(str(output) + ".conversion.json")
    report.write_text(json.dumps({
        "recipe_id": "inkling2_mini-nvfp4-v1",
        "source": {
            "repository": "aday777/Inkling-2-Mini-Abliterated-NVFP4",
            "revision": "40bf42fdff1dc2d619ea01d564f1597363216ebb",
            "model_path": str(Path(model_dir).resolve()),
        },
        "objects": len(specs),
        "bytes": output.stat().st_size,
        "elapsed_seconds": elapsed,
        "notes": [
            "The four MXFP8 expert layers (20-23) are stored as BF16: the artifact "
            "has no UE8M0 block-scale layout, so dequantizing is the exact option "
            "and costs about 4 GB over the packed form.",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    convert(args.model, args.out)


if __name__ == "__main__":
    main()
