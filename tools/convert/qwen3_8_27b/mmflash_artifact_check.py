"""Check the MMFlash draft package inside a produced NInfer artifact.

The MMFlash draft objects carry the whole speculative-decoding path, so a
damaged package shows up as an acceptance rate near zero rather than as a
load error.  This compares every draft object in the artifact against the
checkpoint it was converted from and reports where the two disagree.

Run with the conversion virtualenv:

    /home/chenjx/ninfer-conv-venv/bin/python3 mmflash-artifact-check.py \\
        <artifact.ninfer> /data/models/FSCCS-Qwen3.8-27B-MMFlash
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/chenjx/ninfer-mmflash")

from tools.artifact.layouts import dequantize_row_split  # noqa: E402
from tools.artifact.numeric import W8G32_F16S  # noqa: E402
from tools.convert.common.safetensors import ShardReader  # noqa: E402


def read_v3_directory(path: Path) -> dict:
    with path.open("rb") as handle:
        magic, length = struct.unpack("<8sQ", handle.read(16))
        if magic != b"NINFER\x00\x03":
            raise SystemExit(f"not a v3 container: {magic!r}")
        handle.read(16)  # fingerprint
        return json.loads(handle.read(length).decode("utf-8"))


def read_v2_directory(path: Path) -> dict:
    with path.open("rb") as handle:
        magic, length = struct.unpack("<8sQ", handle.read(16))
        if magic != b"NINFER\x00\x02":
            raise SystemExit(f"not a v2 container: {magic!r}")
        return json.loads(handle.read(length).decode("utf-8"))


def payload_start(path: Path, directory: dict) -> int:
    """Where the payload begins: the directory is padded to a 4096-byte boundary."""
    with path.open("rb") as handle:
        magic, length = struct.unpack("<8sQ", handle.read(16))
    header = 32 if magic.endswith(b"") else 16
    return (header + length + 4095) // 4096 * 4096


def load_object(path: Path, obj: dict, start: int) -> bytes:
    """Read one object.  A v3 object's offset is relative to the payload."""
    with path.open("rb") as handle:
        handle.seek(start + obj["offset"])
        return handle.read(obj["bytes"])


def main() -> int:
    artifact = Path(sys.argv[1] if len(sys.argv) > 1 else
                    "/data/models/heretic_v2_27b_nvfp4_q6emb_q6head_mmflash.ninfer")
    checkpoint = Path(sys.argv[2] if len(sys.argv) > 2 else
                      "/data/models/FSCCS-Qwen3.8-27B-MMFlash")

    with artifact.open("rb") as handle:
        magic = handle.read(8)
    directory = (read_v3_directory(artifact) if magic.endswith(b"\x03")
                 else read_v2_directory(artifact))
    objects = {obj["id"]: obj for obj in directory["objects"]}
    draft = {name: obj for name, obj in objects.items() if name.startswith("dflash/")}
    print(f"artifact  : {artifact}")
    print(f"draft objects: {len(draft)}")

    component = directory.get("components", {}).get("mmflash")
    if component is not None:
        config = component.get("config", {})
        print(f"taps      : {config.get('dflash_config', {}).get('target_layer_ids')}")
        print(f"block size: {config.get('block_size')}")

    start = payload_start(artifact, directory)
    reader = ShardReader.from_file(str(checkpoint / "model.safetensors"))
    try:
        # The norms are BF16 and must match the checkpoint byte for byte.
        print()
        print("norms (BF16, must match the checkpoint exactly):")
        norms = [
            ("dflash/context_norm", "hidden_norm.weight"),
            ("dflash/final_norm", "norm.weight"),
            ("dflash/layers/0/input_norm", "layers.0.input_layernorm.weight"),
            ("dflash/layers/0/attention/query_norm", "layers.0.self_attn.q_norm.weight"),
        ]
        ok = True
        for name, source in norms:
            obj = objects.get(name)
            if obj is None:
                print(f"  {name:<44} MISSING")
                ok = False
                continue
            stored = load_object(artifact, obj, start)
            expected = (reader.get(source).to(torch.bfloat16).contiguous()
                        .view(torch.uint16).numpy().tobytes())
            # The payload may sit at a shifted offset inside the object.
            shift = stored.find(expected)
            if stored == expected:
                print(f"  {name:<44} exact")
            elif shift >= 0:
                print(f"  {name:<44} SHIFTED: real data at +{shift}")
                ok = False
            else:
                print(f"  {name:<44} MISMATCH (data not found in the object)")
                ok = False

        # One projection: the payload must decode to the source within the
        # W8 error bound.
        print()
        print("projections (W8, decoded against the checkpoint):")
        projections = [
            ("dflash/feature_projection", "fc.weight", None),
            ("dflash/layers/0/attention/query_key_value", None,
             ("layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
              "layers.0.self_attn.v_proj.weight")),
            ("dflash/layers/0/mlp/gate_up", None,
             ("layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight")),
        ]
        for name, single, parts in projections:
            obj = objects.get(name)
            if obj is None:
                print(f"  {name:<44} MISSING")
                ok = False
                continue
            payload = load_object(artifact, obj, start)
            if single is not None:
                source = reader.get(single).to(torch.float32).numpy()
            else:
                source = np.concatenate(
                    [reader.get(part).to(torch.float32).numpy() for part in parts], axis=0
                )
            rows, cols = obj["shape"]
            try:
                decoded = dequantize_row_split(payload, W8G32_F16S, (rows, cols),
                                               device="cpu").to(torch.float32).numpy()
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"  {name:<44} UNDECODABLE: {error}")
                ok = False
                continue
            if not np.isfinite(decoded).all():
                print(f"  {name:<44} NOT FINITE (the payload is not in this format)")
                ok = False
                continue
            error = float(np.abs(decoded - source).max())
            relative = error / float(np.abs(source).max())
            status = "ok" if relative < 0.01 else "MISMATCH"
            print(f"  {name:<44} {status} max_abs={error:.5f} rel={relative:.5f}")
            if status != "ok":
                ok = False

        print()
        print("RESULT:", "the draft package is intact" if ok else
              "the draft package is DAMAGED - do not trust the acceptance rate")
        return 0 if ok else 1
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
