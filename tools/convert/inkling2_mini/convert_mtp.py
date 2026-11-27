"""Package the official Inkling-2-Mini BF16 MTP checkpoint as a native NInfer v2 artifact.

All 54 source tensors are already contiguous BF16, so this converter validates the
safetensors directory and streams its payloads without decoding or reallocating them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Iterator

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec


def _payload(source, begin: int, size: int) -> Iterator[bytes]:
    source.seek(begin)
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, 8 * 1024 * 1024))
        if not chunk:
            raise ValueError("truncated safetensors payload")
        yield chunk
        remaining -= len(chunk)


def convert(source_path: Path, output_path: Path) -> None:
    digest = hashlib.sha256()
    with source_path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes == 0 or header_bytes > 100 * 1024 * 1024:
            raise ValueError("invalid safetensors header size")
        digest.update(prefix)
        header_raw = source.read(header_bytes)
        if len(header_raw) != header_bytes:
            raise ValueError("truncated safetensors header")
        digest.update(header_raw)
        header = json.loads(header_raw)
        payload_begin = 8 + header_bytes
        source_size = source_path.stat().st_size
        tensors = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if entry.get("dtype") != "BF16":
                raise ValueError(f"{name}: expected BF16, got {entry.get('dtype')}")
            shape = tuple(entry["shape"])
            if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
                raise ValueError(f"{name}: invalid shape")
            begin, end = entry["data_offsets"]
            size = 2
            for dim in shape:
                size *= dim
            if type(begin) is not int or type(end) is not int or begin < 0 or end - begin != size:
                raise ValueError(f"{name}: invalid data offsets")
            if payload_begin + end > source_size:
                raise ValueError(f"{name}: payload extends beyond file")
            tensors.append((name, shape, payload_begin + begin, size))
        if len(tensors) != 54:
            raise ValueError(f"expected 54 MTP tensors, found {len(tensors)}")
        tensors.sort(key=lambda item: item[2])
        cursor = payload_begin
        for name, _, begin, size in tensors:
            if begin != cursor:
                raise ValueError(f"{name}: source payload is not tightly packed in directory order")
            cursor = begin + size
        if cursor != source_size:
            raise ValueError("unexpected bytes after the MTP tensor payloads")

        specs = [TensorSpec(name, shape, "BF16", "contiguous-le-v1") for name, shape, _, _ in tensors]
        # Artifact objects are emitted in source order; the C++ binder addresses them by name.
        identity = ArtifactIdentity("inkling2-mini-mtp", "thinkingmachines/Inkling-2-Mini/mtp.safetensors")
        with ArtifactWriter(output_path, identity, specs) as writer:
            for name, _, begin, size in tensors:
                writer.write(name, _payload(source, begin, size))
    with Artifact(output_path) as artifact:
        if len(artifact.objects) != 54:
            raise ValueError("packaged artifact does not contain 54 objects")
        if sum(obj.bytes for obj in artifact.objects) != source_size - payload_begin:
            raise ValueError("packaged artifact tensor bytes differ from the source")
    print(json.dumps({"source": str(source_path), "output": str(output_path),
                      "objects": len(tensors), "bytes": output_path.stat().st_size}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.out)


if __name__ == "__main__":
    main()
