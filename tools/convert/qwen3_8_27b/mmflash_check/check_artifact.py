"""Check the MMFlash draft objects in the produced artifact against the checkpoint.

The artifact stores the draft projections as W8G32_F16S (row-split-k128-v1).
This decodes them and compares against the BF16 source, so a broken conversion
shows up as a numerical mismatch rather than as a mysterious acceptance rate.
"""

import json
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/chenjx/ninfer-mmflash")

from tools.artifact.layouts import dequantize_row_split  # noqa: E402
from tools.artifact.numeric import W8G32_F16S  # noqa: E402
from tools.convert.common.safetensors import ShardReader  # noqa: E402

ARTIFACT = "/data/models/heretic_v2_27b_nvfp4_q6emb_q6head_mmflash.ninfer"
CHECKPOINT = "/data/models/FSCCS-Qwen3.8-27B-MMFlash"


def read_directory(path):
    with open(path, "rb") as handle:
        magic, length = struct.unpack("<8sQ", handle.read(16))
        assert magic == b"NINFER\x00\x03", magic
        handle.read(16)  # fingerprint
        return json.loads(handle.read(length).decode("utf-8"))


def decode_w8(payload, rows, cols):
    """Decode row-split-k128-v1 W8G32_F16S into a float32 matrix."""
    from tools.artifact.layouts import dequantize_row_split

    tensor = dequantize_row_split(payload, W8G32_F16S, (rows, cols), device="cpu")
    return tensor.to(torch.float32).numpy()


def main():
    directory = read_directory(ARTIFACT)
    objects = {obj["id"]: obj for obj in directory["objects"]}
    draft = {name: obj for name, obj in objects.items() if name.startswith("dflash/")}
    print("dflash objects in the artifact:", len(draft))

    component = directory["components"].get("mmflash")
    if component is not None:
        config = component.get("config", {})
        print("component config taps:", config.get("dflash_config", {}).get("target_layer_ids"))

    reader = ShardReader.from_file(f"{CHECKPOINT}/model.safetensors")
    try:
        # The feature projection mixes the five target taps: fc.weight [5120, 25600].
        obj = draft["dflash/feature_projection"]
        rows, cols = obj["shape"]
        assert obj["format"] in ("W8G32_F16S", "q8_g32_fp16"), obj["format"]
        with open(ARTIFACT, "rb") as handle:
            handle.seek(obj["offset"])
            payload = handle.read(obj["bytes"])
        decoded = decode_w8(payload, rows, cols)
        source = reader.get("fc.weight").to(torch.float32).numpy()
        err = np.abs(decoded - source)
        print()
        print("feature_projection [5120, 25600]")
        print("  max abs error :", float(err.max()))
        print("  relative max  :", float(err.max() / np.abs(source).max()))
        print("  cosine        :", float(
            np.dot(decoded.ravel(), source.ravel())
            / (np.linalg.norm(decoded) * np.linalg.norm(source))
        ))

        # One attention projection of one layer.
        obj = draft["dflash/layers/0/attention/query_key_value"]
        rows, cols = obj["shape"]
        with open(ARTIFACT, "rb") as handle:
            handle.seek(obj["offset"])
            payload = handle.read(obj["bytes"])
        decoded = decode_w8(payload, rows, cols)
        q = reader.get("layers.0.self_attn.q_proj.weight").to(torch.float32).numpy()
        k = reader.get("layers.0.self_attn.k_proj.weight").to(torch.float32).numpy()
        v = reader.get("layers.0.self_attn.v_proj.weight").to(torch.float32).numpy()
        source = np.concatenate([q, k, v], axis=0)
        err = np.abs(decoded - source)
        print()
        print("layers/0/attention/query_key_value [6144, 5120]")
        print("  max abs error :", float(err.max()))
        print("  relative max  :", float(err.max() / np.abs(source).max()))
        print("  row-0 matches q_proj:", bool(np.allclose(decoded[:4096], q, atol=0.02)))
        print("  row-4096 matches k_proj:", bool(np.allclose(decoded[4096:5120], k, atol=0.02)))
        print("  row-5120 matches v_proj:", bool(np.allclose(decoded[5120:], v, atol=0.02)))

        # The gate/up rows.
        obj = draft["dflash/layers/0/mlp/gate_up"]
        rows, cols = obj["shape"]
        with open(ARTIFACT, "rb") as handle:
            handle.seek(obj["offset"])
            payload = handle.read(obj["bytes"])
        decoded = decode_w8(payload, rows, cols)
        gate = reader.get("layers.0.mlp.gate_proj.weight").to(torch.float32).numpy()
        up = reader.get("layers.0.mlp.up_proj.weight").to(torch.float32).numpy()
        source = np.concatenate([gate, up], axis=0)
        err = np.abs(decoded - source)
        print()
        print("layers/0/mlp/gate_up [34816, 5120]")
        print("  max abs error :", float(err.max()))
        print("  relative max  :", float(err.max() / np.abs(source).max()))
        print("  rows 0..17407 match gate:", bool(np.allclose(decoded[:17408], gate, atol=0.02)))
        print("  rows 17408.. match up  :", bool(np.allclose(decoded[17408:], up, atol=0.02)))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
