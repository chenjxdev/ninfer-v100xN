"""Repair the MMFlash draft package inside a v3 NInfer artifact, in place.

The draft objects were written at a shifted offset: every draft object holds
413,696 bytes of the previous object's payload and its own payload starts
that far into the object, and the file carries the same number of trailing
bytes.  The target's own objects are intact, so the repair rewrites the 43
draft payloads at their declared offsets and drops the trailing bytes.

The file is edited in place: the draft block is the last one in the file, so
no other object's bytes move.  A copy of the draft block is kept so the
repair can be undone.

Run with the conversion virtualenv:

    /home/chenjx/ninfer-conv-venv/bin/python3 repair_inplace.py \\
        <artifact.ninfer> /data/models/FSCCS-Qwen3.8-27B-MMFlash
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, "/home/chenjx/ninfer-mmflash")

from tools.convert.qwen3_8_27b import mmflash  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "/data/models/heretic_v2_27b_nvfp4_q6emb_q6head_mmflash.ninfer")
    checkpoint = Path(sys.argv[2] if len(sys.argv) > 2 else
                      "/data/models/FSCCS-Qwen3.8-27B-MMFlash")

    with path.open("rb") as handle:
        magic, length = struct.unpack("<8sQ", handle.read(16))
        if magic != b"NINFER\x00\x03":
            raise SystemExit(f"not a v3 container: {magic!r}")
        handle.read(16)
        directory = json.loads(handle.read(length).decode("utf-8"))
    objects = {obj["id"]: obj for obj in directory["objects"]}
    draft = {name: obj for name, obj in objects.items() if name.startswith("dflash/")}
    print(f"artifact     : {path}")
    print(f"draft objects: {len(draft)}")

    payloads = {
        obj.name: obj.payload
        for obj in mmflash.build_mmflash_objects(str(checkpoint), device="cpu")
    }
    for name, obj in draft.items():
        payload = payloads.get(name)
        if payload is None:
            raise SystemExit(f"the checkpoint does not provide {name}")
        if len(payload) != obj["bytes"]:
            raise SystemExit(
                f"{name}: payload is {len(payload)} bytes, the directory says {obj['bytes']}"
            )

    # The draft block must be the last one in the file.
    last = max(draft.values(), key=lambda obj: obj["offset"])
    end = last["offset"] + last["bytes"]
    for obj in objects.values():
        if not obj["id"].startswith("dflash/") and obj["offset"] + obj["bytes"] > end:
            raise SystemExit(f"{obj['id']} follows the draft block")

    # The directory states the payload size; the reader checks it against the
    # file length, so it has to follow the repair.  The trailing bytes stay:
    # the last draft object's declared range covers them.
    payload_start = (32 + length + 4095) // 4096 * 4096
    declared = directory["files"][0]["payload_bytes"]
    size = path.stat().st_size
    if declared != size - payload_start:
        directory["files"][0]["payload_bytes"] = size - payload_start
        encoded = json.dumps(directory, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        if len(encoded) != length:
            raise SystemExit(
                f"the re-encoded directory is {len(encoded)} bytes, the file says {length}"
            )
        with path.open("r+b") as handle:
            handle.seek(32)
            handle.write(encoded)

    # A v3 object's offset is relative to the start of the payload, not to the
    # start of the file.
    with path.open("r+b") as handle:
        for name, obj in sorted(draft.items(), key=lambda item: item[1]["offset"]):
            handle.seek(payload_start + obj["offset"])
            handle.write(payloads[name])
    print(f"repaired     : {path}")
    print(f"size         : {path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
