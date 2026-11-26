"""Build the Qwen3.8-27B mixed NVFP4/FP8 artifact from a ModelOpt checkpoint.

Canonical invocations::

    python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
      --model /data/models/nvidia-qad \
      --out /data/models/qwen3_8_27b_q4km_v2.ninfer

    NINFER_NVFP4_SOURCE=daisyrose python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
      --model /data/models/daisyrose-27b-nvfp4 \
      --out /data/models/daisyrose_27b_nvfp4_q6head_q4mtp.ninfer

The source profile (tools.convert.qwen3_8_27b.source_profile) selects the
checkpoint and its expectations; it is read from NINFER_NVFP4_SOURCE before
this module is imported.

One quantized source, one artifact, no patching round: the 204 NVFP4 text
projections are byte passthroughs (a QAD weight_scale_2 omitted for a
multiplier of 1.0 reads as exactly 1.0) and the 52 FP8 attention objects
pass through as row-scaled FP8, per-row BF16 scales broadcast from the
per-tensor F32 weight_scale -- code words stay bit-identical, no
requantization. A source that exported a layer as ModelOpt FP8 but the
artifact stores as NVFP4 (DaisyRose exports every GDN layer that way, the
production profile keeps thirty-eight of them as NVFP4) is dequantized and
requantized to NVFP4 instead. Unlike the Huihui round the lm_head is itself
NVFP4-quantized: QadSourceReader dequantizes it to BF16
(W = e2m1 * f32(weight_scale) * weight_scale_2, engine nibble order)
before grouped quantization and the draft-head row gather. The QAD export
ships only the 11 MTP layer tensors, so the four remaining MTP objects are
read from the auxiliary Huihui checkpoint; a source that ships them itself
needs no auxiliary checkpoint. Frontend resources are taken verbatim from the
source directory. The artifact lands in the mixed ``Qwen38Nvfp4`` weights
profile (weights_id "nvfp4").
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.artifact.container_v3 import V3ArtifactWriter
from tools.artifact.layouts import (
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader, TensorMetadata
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head

from . import inventory_nvfp4 as inventory
from . import mmflash
from . import v3_plan
from . import recipe_nvfp4 as recipe
from . import source_profile


# Each source profile is a closed recipe: its own recipe id, its own output
# name and its own MTP source. The artifact's name must match the one the
# conversion accepts.
RECIPE_IDS = {
    "qad": "qwen3_8_27b_nvfp4-v5-nvidia-qad-q4mtp-q6emb-w8head",
    "daisyrose": "qwen3_8_27b_nvfp4-v5-radixark-daisyrose-27b-q4mtp-q6emb-q6head",
}
OUTPUT_BASENAMES = {
    "qad": "qwen3_8_27b_q4km_v2.ninfer",
    "daisyrose": "daisyrose_27b_nvfp4_q6emb_q6head_q4mtp.ninfer",
}
RECIPE_ID = RECIPE_IDS[source_profile.ACTIVE.name]
OUTPUT_BASENAME = OUTPUT_BASENAMES[source_profile.ACTIVE.name]

# The QAD export ships only the 11 MTP layer tensors; these four BF16
# objects are read verbatim from the auxiliary Huihui checkpoint. A source
# that ships them itself (DaisyRose) needs no auxiliary checkpoint.
AUXILIARY_MTP_REPOSITORY = source_profile.ACTIVE.mtp_auxiliary
AUXILIARY_MTP_FIELDS = source_profile.ACTIVE.mtp_auxiliary_fields
LM_HEAD_WEIGHT = "lm_head.weight"
LM_HEAD_QUAD_FIELDS = (
    LM_HEAD_WEIGHT,
    "lm_head.weight_scale",
    "lm_head.weight_scale_2",
    "lm_head.input_scale",
)
LM_HEAD_ROWS, LM_HEAD_COLS = 248320, 5120

# Signed E2M1 values indexed by the raw 4-bit word (sign bit 0x8), matching
# the engine's __nv_fp4x2_e2m1 decode: low nibble = even k, high = odd k.
_E2M1_SIGNED = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def dequantize_nvfp4_matrix(
    packed: torch.Tensor,
    scales: torch.Tensor,
    scale_2: float,
    chunk_rows: int = 8192,
) -> torch.Tensor:
    """W = e2m1(code) * f32(weight_scale) * scale_2 as a BF16 [N, K] matrix.

    Matches the engine kernels exactly: they evaluate
    ``e2m1 * scale * inverse_weight_divisor`` with the artifact storing
    divisor = 1 / weight_scale_2, so the checkpoint multiplier applies as-is.
    """

    rows, half_cols = tuple(packed.shape)
    cols = half_cols * 2
    lut = torch.tensor(_E2M1_SIGNED, dtype=torch.float32)
    out = torch.empty((rows, cols), dtype=torch.bfloat16)
    for begin in range(0, rows, chunk_rows):
        end = min(begin + chunk_rows, rows)
        codes = packed[begin:end].to(torch.int64)
        values = torch.empty((end - begin, cols), dtype=torch.float32)
        values[:, 0::2] = lut[codes & 0xF]
        values[:, 1::2] = lut[codes >> 4]
        block = scales[begin:end].to(torch.float32)
        values *= block.repeat_interleave(16, dim=1)
        values *= scale_2
        out[begin:end] = values.to(torch.bfloat16)
    return out


class QadSourceReader:
    """ShardReader overlay presenting the QAD checkpoint as the recipe source.

    lm_head is NVFP4-quantized in QAD: the four raw quad fields are hidden
    and ``lm_head.weight`` resolves to the dequantized BF16 matrix (cached)
    consumed by the W8 output head and the draft-head row gather. The four
    MTP tensors missing from the QAD export resolve to the auxiliary Huihui
    checkpoint. Everything else delegates to the wrapped QAD reader.
    """

    def __init__(
        self,
        quantized_dir: str | Path,
        auxiliary_dir: str | Path | None = AUXILIARY_MTP_REPOSITORY,
    ) -> None:
        self._reader = ShardReader(quantized_dir)
        self._auxiliary_dir = Path(auxiliary_dir) if auxiliary_dir else None
        self._auxiliary = None
        self._auxiliary_metadata = None
        self._lm_head = None

    @property
    def names(self):
        base = set(self._reader.names)
        base.difference_update(LM_HEAD_QUAD_FIELDS)
        base.add(LM_HEAD_WEIGHT)
        if self._auxiliary_dir is not None:
            base.update(AUXILIARY_MTP_FIELDS)
        return tuple(sorted(base))

    def has(self, name: str) -> bool:
        if self._auxiliary_dir is not None and name in AUXILIARY_MTP_FIELDS:
            return True
        if name in LM_HEAD_QUAD_FIELDS:
            return name == LM_HEAD_WEIGHT
        return self._reader.has(name)

    def get(self, name: str) -> torch.Tensor:
        if name == LM_HEAD_WEIGHT:
            if self._lm_head is None:
                self._lm_head = self._materialize_lm_head()
            return self._lm_head
        if self._auxiliary_dir is not None and name in AUXILIARY_MTP_FIELDS:
            if self._auxiliary is None:
                self._auxiliary = ShardReader(self._auxiliary_dir)
            return self._auxiliary.get(name)
        return self._reader.get(name)

    def metadata(self, names):
        plain = [
            name
            for name in names
            if name not in LM_HEAD_QUAD_FIELDS and self._reader.has(name)
        ]
        result = dict(self._reader.metadata(plain)) if plain else {}
        virtual = [name for name in names if name not in result]
        if virtual:
            if self._auxiliary_metadata is None and self._auxiliary_dir is not None:
                if self._auxiliary is None:
                    self._auxiliary = ShardReader(self._auxiliary_dir)
                self._auxiliary_metadata = self._auxiliary.metadata(
                    AUXILIARY_MTP_FIELDS
                )
            for name in virtual:
                if name == LM_HEAD_WEIGHT:
                    result[name] = TensorMetadata(
                        name=name,
                        shard="(nvfp4-dequant)",
                        shape=(LM_HEAD_ROWS, LM_HEAD_COLS),
                        dtype="BF16",
                    )
                elif name in AUXILIARY_MTP_FIELDS and self._auxiliary_metadata is not None:
                    result[name] = self._auxiliary_metadata[name]
                else:
                    raise KeyError(name)
        return result

    def _materialize_lm_head(self) -> torch.Tensor:
        packed = self._reader.get("lm_head.weight")
        scales = self._reader.get("lm_head.weight_scale")
        raw_multiplier = self._reader.get("lm_head.weight_scale_2")
        if (
            packed.dtype != torch.uint8
            or tuple(packed.shape) != (LM_HEAD_ROWS, LM_HEAD_COLS // 2)
            or scales.dtype != torch.float8_e4m3fn
            or tuple(scales.shape) != (LM_HEAD_ROWS, LM_HEAD_COLS // 16)
            or raw_multiplier.dtype != torch.float32
            or raw_multiplier.numel() != 1
        ):
            raise ValueError("lm_head: QAD NVFP4 quad signature mismatch")
        scale_2 = float(raw_multiplier.reshape(()).item())
        if not (scale_2 > 0.0 and scale_2 < float("inf")):
            raise ValueError("lm_head: weight_scale_2 is not finite positive")
        matrix = dequantize_nvfp4_matrix(packed, scales, scale_2)
        del packed, scales, raw_multiplier
        print(
            f"lm_head dequantized: {tuple(matrix.shape)} BF16, "
            f"weight_scale_2={scale_2:.6g}",
            flush=True,
        )
        return matrix

    def close(self) -> None:
        self._reader.close()
        if self._auxiliary is not None:
            self._auxiliary.close()
        self._lm_head = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    quantized_dir: Path
    config_summary: dict[str, object]
    quantized_source: family_conversion.SourcePreflight | object
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_index(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    value = family_conversion.load_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map must be a nonempty object")
    referenced = set(weight_map.values())
    actual = {path.name for path in model_dir.glob("*.safetensors")}
    if actual != referenced:
        raise ValueError(f"{model_dir}: safetensors shard set does not match the index")
    for shard in sorted(referenced):
        path = model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{path}: indexed shard is missing or empty")


def load_resources(model_dir: Path) -> tuple[family_conversion.ResourcePayload, ...]:
    """Load frontend resources verbatim (Huihui tokenizer is unofficial).

    One normalization: ModelOpt-stripped tokenizer_config.json may omit
    ``add_bos_token``; the engine validates the Qwen prefix semantics and
    defaults a missing flag to ``true``. Qwen tokenizers never prepend BOS,
    so the explicit ``false`` restores the checkpoint's real behavior.
    """

    import json

    expected_names = tuple(spec.name for spec in inventory.RESOURCE_SPECS)
    resources = list(
        family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)
    )
    actual_names = tuple(resource.name for resource in resources)
    if actual_names != expected_names:
        raise ValueError(
            f"frontend resource set mismatch: expected {expected_names!r}, "
            f"got {actual_names!r}"
        )
    for index, resource in enumerate(resources):
        if resource.name != "frontend/tokenizer_config.json":
            continue
        config = json.loads(resource.data)
        changed = False
        if "add_bos_token" not in config:
            config["add_bos_token"] = False
            changed = True
        if "add_prefix_space" not in config:
            config["add_prefix_space"] = False
            changed = True
        if "added_tokens_decoder" not in config:
            # ModelOpt strips the decoder map; rebuild it from tokenizer.json's
            # added_tokens list (the authoritative special-token table).
            tokenizer = json.loads(
                (model_dir / "tokenizer.json").read_bytes()
            )
            decoder = {
                str(token["id"]): {
                    "content": token["content"],
                    "lstrip": token.get("lstrip", False),
                    "rstrip": token.get("rstrip", False),
                    "normalized": token.get("normalized", False),
                    "special": token.get("special", True),
                }
                for token in tokenizer.get("added_tokens", [])
            }
            for token in decoder.values():
                token["single_word"] = False
            if not decoder:
                raise ValueError("tokenizer.json has no added_tokens to rebuild")
            config["added_tokens_decoder"] = decoder
            changed = True
        if changed:
            data = json.dumps(
                config, ensure_ascii=False, indent=2
            ).encode("utf-8") + b"\n"
            resources[index] = family_conversion.ResourcePayload(
                resource.name, data
            )
    return tuple(resources)


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(
    quantized_dir: str | Path,
    auxiliary_dir: str | Path = AUXILIARY_MTP_REPOSITORY,
) -> ConversionPreflight:
    quantized = Path(quantized_dir)
    _validate_index(quantized)

    config = family_conversion.load_json(quantized / "config.json")
    quantization = config.get("quantization_config", {})
    if quantization.get("quant_method") not in (None, "modelopt", "quantize"):
        # ModelOpt checkpoints may carry no quantization_config at all in
        # config.json (hf_quant_config.json is authoritative); accept both.
        raise ValueError(
            f"unexpected quant_method {quantization.get('quant_method')!r}"
        )
    quant_config = family_conversion.load_json(quantized / "hf_quant_config.json")
    if quant_config.get("producer", {}).get("name") != "modelopt":
        raise ValueError("hf_quant_config.json producer is not modelopt")
    # NVIDIA QAD exports declare MIXED_PRECISION (per-layer recipe in
    # exclude_modules); plain NVFP4 QAT checkpoints declare NVFP4.
    if quant_config.get("quantization", {}).get("quant_algo") not in (
        "NVFP4",
        "MIXED_PRECISION",
    ):
        raise ValueError("hf_quant_config.json quant_algo is not NVFP4")
    config_summary = family_config.validate_config(config)
    preflight_inventory()

    with QadSourceReader(quantized, auxiliary_dir) as quantized_reader:
        quantized_source = recipe.preflight_quantized_metadata(quantized_reader)

    resources = load_resources(quantized)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, quantized)
    return ConversionPreflight(
        quantized_dir=quantized,
        config_summary=config_summary,
        quantized_source=quantized_source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def _encode_nvfp4_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> bytes:
    selected = recipe.NVFP4_WEIGHTS_BY_NAME[spec.name]
    packed, scales, divisor = recipe.materialize_nvfp4_weight(selected, reader)
    return encode_nvfp4(packed, scales, divisor, spec.shape)


def _encode_fp8_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> bytes:
    selected = recipe.FP8_WEIGHTS_BY_NAME[spec.name]
    codes, row_scales = recipe.materialize_fp8_weight(selected, reader)
    return encode_fp8_row_scaled(codes, row_scales, spec.shape)


def _encode_grouped_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    device: torch.device,
) -> bytes:
    selected = recipe.GROUPED_WEIGHTS_BY_NAME[spec.name]
    fused = recipe.materialize_grouped_weight(selected, reader)
    payload = family_conversion.encode_tensor_payload(fused, spec, device)
    del fused
    return payload


def _materialize_direct(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> torch.Tensor:
    tensor = recipe.materialize_direct(spec.name, reader)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def _materialize_derived(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    derived: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    tensor = recipe.materialize_derived(spec.name, reader, dict(derived))
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    auxiliary_dir: str | Path,
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.quantized_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.quantized_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["source"] = {
        "quantized": {
            "repository": recipe.QUANTIZED_REPOSITORY,
            "revision": recipe.QUANTIZED_REVISION,
            "model_path": str(preflight.quantized_dir.resolve()),
        },
        "auxiliary_mtp": {
            "model_path": (
                str(Path(auxiliary_dir).resolve()) if auxiliary_dir else None
            ),
            "fields": list(AUXILIARY_MTP_FIELDS),
        },
        "ranking_path": str(ranking.resolve()),
    }
    return report


def _make_writer(output: Path, specs):
    """The artifact writer the source profile asks for.

    The v3 container carries the logical-parameter bindings, the activation
    input divisor of every NVFP4 projection and the per-component resources.
    That is what lets the container hold a draft package, such as the
    first-generation DFlash draft (MMFlash), with the configuration the
    engine validates it against.
    """
    identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    if source_profile.ACTIVE.container_version == "v3":
        return V3ArtifactWriter(
            output, identity, specs, v3_plan.build_plan(specs)
        )
    return ArtifactWriter(output, identity, specs)


def convert(
    quantized_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    auxiliary_dir: str | Path = AUXILIARY_MTP_REPOSITORY,
    mmflash_model: str | Path | None = None,
) -> Path:
    """Run the closed single-source conversion and return its report path."""

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"NVFP4 converter output basename must be {OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(quantized_dir, auxiliary_dir)

    print(
        f"source profile: {source_profile.ACTIVE.name} "
        f"({source_profile.ACTIVE.repository} @ "
        f"{source_profile.ACTIVE.revision})",
        flush=True,
    )
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.NVFP4_SOURCES)} NVFP4 quads "
        f"({len(recipe.REQUANTIZED_SOURCES)} requantized from FP8), "
        f"{len(recipe.FP8_SOURCES)} FP8 quads and "
        f"{len(recipe.GROUPED_SOURCES)} grouped BF16 sources, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    # The first-generation DFlash draft (MMFlash) ships as its own
    # checkpoint, so its objects are built from that checkpoint rather than
    # from the target's source.
    mmflash_payloads: dict[str, bytes] = {}
    if mmflash_model is not None:
        for obj in mmflash.build_mmflash_objects(mmflash_model, device=resolved_device):
            mmflash_payloads[obj.name] = obj.payload
    with QadSourceReader(
        preflight.quantized_dir, auxiliary_dir
    ) as quantized_reader:
        with _make_writer(output, preflight.object_plan.specs) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "writer object plan differs from completed preflight"
                )
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                payload: bytes | Iterable[bytes]
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name in mmflash_payloads:
                    payload = mmflash_payloads[spec.name]
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    payload = _encode_nvfp4_weight(spec, quantized_reader)
                elif spec.name in recipe.FP8_WEIGHTS_BY_NAME:
                    payload = _encode_fp8_weight(spec, quantized_reader)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_input_divisor(
                        recipe.INPUT_DIVISORS_BY_NAME[spec.name],
                        quantized_reader,
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.GROUPED_WEIGHTS_BY_NAME:
                    payload = _encode_grouped_weight(spec, quantized_reader, resolved_device)
                elif spec.name in recipe.DIRECT_BY_NAME:
                    tensor = _materialize_direct(spec, quantized_reader)
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                else:
                    tensor = _materialize_derived(spec, quantized_reader, derived)
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                if index % 16 == 0 or index == len(inventory.OBJECT_SPECS):
                    print(
                        f"[{index}/{len(inventory.OBJECT_SPECS)}] written",
                        flush=True,
                    )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(quantized_dir),
        "out": str(out_path),
        "device": requested_device,
        "mtp_source": str(auxiliary_dir) if auxiliary_dir else None,
        "mmflash_source": str(mmflash_model) if mmflash_model else None,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        auxiliary_dir=auxiliary_dir,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--source",
        default=source_profile.ACTIVE.name,
        help=(
            "source profile; the profile is selected by NINFER_NVFP4_SOURCE "
            "before this module is imported, so this argument only reports "
            "the active one"
        ),
    )
    parser.add_argument(
        "--mtp-source",
        default=AUXILIARY_MTP_REPOSITORY,
        help="auxiliary checkpoint providing the four MTP tensors QAD omits",
    )
    parser.add_argument(
        "--mmflash-model",
        default=None,
        type=Path,
        help=(
            "checkpoint providing the first-generation DFlash draft (MMFlash); "
            "without it the artifact carries no draft package"
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.source != source_profile.ACTIVE.name:
        raise SystemExit(
            f"--source {arguments.source!r} does not match the active profile "
            f"{source_profile.ACTIVE.name!r}; set "
            f"NINFER_NVFP4_SOURCE={arguments.source} before importing the "
            "converter"
        )
    convert(
        arguments.model,
        arguments.out,
        device=arguments.device,
        auxiliary_dir=arguments.mtp_source,
        mmflash_model=arguments.mmflash_model,
    )


if __name__ == "__main__":
    main()
