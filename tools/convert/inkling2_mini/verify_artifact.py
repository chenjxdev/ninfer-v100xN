"""Verify the Inkling-2-Mini artifact against its source checkpoint.

The routed experts are the only tensors the converter touches beyond a byte
copy: the packed codes and the E4M3 block scales are copied verbatim and the
per-expert multiplier moves into the object's weight divisor. This check
dequantizes both sides the way the engine does and compares them.
"""
import json
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/chenjx/ninfer-jack")
from tools.artifact.container import Artifact
from tools.artifact.layouts import decode_fp8_row_scaled_words, decode_nvfp4_words
from tools.convert.common.safetensors import ShardReader

SRC = "/data/models/Inkling-2-Mini-Abliterated-NVFP4"
ART = "/data/models/inkling2_mini_nvfp4.ninfer"

E2M1 = np.array((0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6), dtype=np.float32)


def e4m3(raw):
    return torch.from_numpy(np.ascontiguousarray(raw).reshape(-1)).view(torch.float8_e4m3fn).float().numpy()


def unpack(codes, scales, rows, cols):
    values = np.empty((rows, cols), dtype=np.float32)
    values[:, 0::2] = E2M1[codes & 15]
    values[:, 1::2] = E2M1[codes >> 4]
    values *= np.repeat(scales, 16, axis=1)
    return values


def main():
    reader = ShardReader(SRC)
    art = Artifact.open(ART)
    try:
        objects = {o.name: o for o in art.objects}
        print("=== 产物结构 ===")
        print("  对象数:", len(objects), " 字节:", art.file_bytes)
        print("  identity:", art.identity)
        kinds = {}
        for o in art.objects:
            kinds[getattr(o, "format", "RESOURCE")] = kinds.get(getattr(o, "format", "RESOURCE"), 0) + 1
        print("  格式分布:", kinds)

        print("\n=== 专家：产物 vs 源（按引擎语义反量化后逐位比较）===")
        for layer, expert in ((0, 0), (0, 77), (6, 13), (19, 5), (11, 100)):
            prefix = "model.llm.layers.%d.mlp.experts" % layer
            for which, cols in (("w13", 2304), ("w2", 1152)):
                name = "text/layers/%d/mlp/experts/%d/%s" % (layer, expert, which)
                obj = objects[name]
                codes, scale_words, divisor = decode_nvfp4_words(
                    art.payload(name), (2304, cols))
                divisor = float(divisor.reshape(()).item())
                artifact_values = unpack(
                    codes.numpy(),
                    e4m3(scale_words.numpy().reshape(-1)).reshape(2304, cols // 16),
                    2304, cols) / divisor

                src_codes = reader.get("%s.%s_weight" % (prefix, which))[expert].numpy()
                src_scales = e4m3(
                    reader.get("%s.%s_weight_scale" % (prefix, which))[expert].view(torch.uint8).numpy().reshape(-1)
                ).reshape(2304, cols // 16)
                amax = float(reader.get("%s.%s_weight_amax" % (prefix, which))[expert])
                denom = float(reader.get(
                    "%s.%s_weight_outer_scale_denominator" % (prefix, which))[0])
                source_values = unpack(src_codes, src_scales, 2304, cols) * (amax / denom)

                delta = np.abs(artifact_values - source_values).max()
                exact = np.array_equal(artifact_values, source_values)
                print("  layer %-2d expert %-3d %-4s divisor=%.6g  最大误差 %.3g  完全相同=%s"
                      % (layer, expert, which, divisor, delta, exact))

        print("\n=== 稠密权重：逐字节相同 ===")
        for name, obj_name in (
            ("model.llm.layers.0.attn.wq_du.weight", "text/layers/0/attn/q"),
            ("model.llm.layers.0.mlp.gate.weight", "text/layers/0/mlp/gate"),
            ("model.llm.embed.weight", "text/embed"),
            ("model.visual.proj_in.weight", "vision/proj_in"),
        ):
            source = reader.get(name)
            payload = art.payload(obj_name)
            same = payload == source.contiguous().view(torch.uint16).numpy().tobytes()
            print("  %-46s -> %-28s 逐字节相同=%s" % (name, obj_name, same))

        print("\n=== Inkling 特定布局：relative-bias 转置 / static taps 反转 ===")
        for name, obj_name, transformed in (
            ("model.llm.layers.0.attn.rel_logits_proj.proj", "text/layers/0/attn/rel_proj",
             reader.get("model.llm.layers.0.attn.rel_logits_proj.proj").T.contiguous()),
            ("model.llm.layers.5.attn.k_sconv.static_w", "text/layers/5/attn/k_sconv_static",
             reader.get("model.llm.layers.5.attn.k_sconv.static_w").flip(0).contiguous()),
        ):
            same = art.payload(obj_name) == transformed.view(torch.uint16).numpy().tobytes()
            print("  %-46s -> %-28s 语义布局相同=%s" % (name, obj_name, same))
            if not same:
                raise AssertionError(obj_name + " has incorrect Inkling-specific layout")

        print("\n=== MXFP8 层（20-23）：原始 codes 和 group scales 是否与源完全一致 ===")
        layer, expert = 23, 42
        prefix = "model.llm.layers.%d.mlp.experts" % layer
        for which, cols in (("w13", 2304), ("w2", 1152)):
            name = "text/layers/%d/mlp/experts/%d/%s" % (layer, expert, which)
            codes, _ = decode_fp8_row_scaled_words(art.payload(name), (2304, cols))
            artifact_codes = codes.numpy()
            group_scales = torch.frombuffer(
                bytearray(art.payload(name + "_mxscale")), dtype=torch.uint16
            ).view(torch.bfloat16).float().numpy().reshape(2304, cols // 32)
            artifact_values = e4m3(artifact_codes).reshape(2304, cols) * np.repeat(group_scales, 32, axis=1)

            source_codes = reader.get("%s.%s_weight" % (prefix, which))[expert].view(torch.uint8).numpy()
            source_ue8m0 = reader.get("%s.%s_weight_scale" % (prefix, which))[expert].numpy().astype(np.int32)
            source_scales = np.ldexp(np.ones_like(source_ue8m0, dtype=np.float32), source_ue8m0 - 127)
            source_values = e4m3(source_codes).reshape(2304, cols) * np.repeat(
                source_scales.reshape(2304, cols // 32), 32, axis=1)
            delta = np.abs(artifact_values - source_values).max()
            exact = np.array_equal(artifact_values, source_values)
            print("  layer %d expert %d %-4s 最大误差 %.3g 完全相同=%s" % (layer, expert, which, delta, exact))
            if not exact:
                raise AssertionError(name + " MXFP8 payload differs from source")
    finally:
        # NumPy/PyTorch can retain exported mmap views until garbage collection;
        # process teardown closes the mapping and must not mask verification results.
        try:
            art.close()
        except BufferError:
            pass
        reader.close()


if __name__ == "__main__":
    main()
