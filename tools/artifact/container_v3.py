"""Streaming writer for the NInfer v3 single-file container.

The v3 container is what the engine's QUASAR adapter reads (see
``src/artifact/reader.cpp``)::

    [0,8)      magic ``NINFER\\x00\\x03``
    [8,16)     u64 directory JSON length
    [16,32)    16-byte container identity
    [32,32+n)  directory JSON
    payload    starts at ``align_up(32 + n, 4096)``

The directory carries five sections:

``objects``
    The physical payloads: id, kind, shape/encoding, format, layout, the
    payload-relative offset and the byte length.
``bindings``
    Logical parameter -> physical object plus the element range it covers.
    A fused object exposes several logical parameters (query/key/gate/value
    of one attention projection); their ranges must tile the object.
``uses``
    The activation input divisor of a logical parameter. The engine looks it
    up by the parameter name and requires exactly one candidate.
``components``
    The frontend resources, split per component.
``files``
    The single-file container description: a null path and the payload size.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field as dataclasses_field
from pathlib import Path
from typing import Iterable, Sequence, TypeAlias

from .container import (
    ArtifactError,
    ArtifactIdentity,
    ArtifactObject,
    ObjectSpec,
    plan_objects,
)
from .layout_metadata import align_up

MAGIC_V3 = b"NINFER\x00\x03"
V3_HEADER_BYTES = 32
PAYLOAD_ALIGNMENT = 4096

#: The container identity the engine registers the published QUASAR artifact
#: by (reader.cpp compares these sixteen bytes at offset 16).
QUASAR_ID = bytes.fromhex("39be2299c2b94512abd51ffe079830f2")
#: The identity of a container that is not the published QUASAR one.
NEUTRAL_ID = bytes(16)

#: The v3 spelling of the v2 numeric formats and layouts.
V3_FORMATS = {
    "BF16": "bf16",
    "FP32": "fp32",
    "I32": "int32",
    "Q4G64_F16S": "q4_g64_fp16",
    "Q5G64_F16S": "q5_g64_fp16",
    "Q6G64_F16S": "q6_g64_fp16",
    "W8G32_F16S": "q8_g32_fp16",
    "NVFP4": "nvfp4",
    "FP8_E4M3FN_ROW_BF16S": "fp8_e4m3fn_row_bf16",
}
V3_LAYOUTS = {
    "contiguous-le-v1": "contiguous_le_v1",
    "row-split-k128-v1": "row_split_k128_v1",
    "blockscale-k16-m128x4-v1": "block_scale_k16_m128x4_v1",
    "row-scale-v1": "row_scale_v1",
}
#: The v3 spelling of the resource encoding.
V3_RESOURCE_ENCODING = {"raw-bytes-v1": "raw_bytes_v1"}

PayloadChunk: TypeAlias = bytes | bytearray | memoryview
Payload: TypeAlias = PayloadChunk | Iterable[PayloadChunk]


@dataclass(frozen=True, slots=True)
class V3Binding:
    """One logical parameter: a physical object and the element range it owns.

    A binding that owns the whole object uses the ``object`` form; one that
    owns a row range of a fused object uses the ``parts`` form, because the
    engine resolves the ``object`` form as the object's full element range.
    """

    name: str
    object: str
    begin: int
    end: int
    elements: int

    def to_json(self) -> dict[str, object]:
        if self.begin == 0 and self.end == self.elements:
            return {"object": self.object}
        return {
            "parts": [
                {"object": self.object, "range": [self.begin, self.end]},
            ]
        }


@dataclass(frozen=True, slots=True)
class V3Use:
    """One logical parameter and the activation input divisor it reads."""

    parameter: str
    divisor: str

    def to_json(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "auxiliaries": {
                "activation_input_divisor": {"object": self.divisor},
            },
        }


@dataclass(frozen=True, slots=True)
class V3Plan:
    """The directory sections the engine resolves parameters through.

    A component names the resources it carries and may also carry a
    ``config`` object: the engine validates the geometry of a draft
    package against that configuration before it binds the package, so the
    writer has to be able to state it.
    """

    bindings: tuple[V3Binding, ...]
    uses: tuple[V3Use, ...]
    components: dict[str, dict[str, dict[str, str]]]
    component_configs: dict[str, dict[str, object]] = dataclasses_field(default_factory=dict)


def _payload_chunks(payload: Payload) -> Iterable[memoryview]:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        yield memoryview(payload).cast("B")
        return
    for chunk in payload:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("payload chunks must support the buffer protocol")
        yield memoryview(chunk).cast("B")


def _object_json(obj: ArtifactObject) -> dict[str, object]:
    if obj.kind == "tensor":
        return {
            "id": obj.name,
            "kind": "tensor",
            "shape": list(obj.shape),
            "format": V3_FORMATS[obj.format],
            "layout": V3_LAYOUTS[obj.layout],
            "offset": obj.offset,
            "bytes": obj.bytes,
        }
    return {
        "id": obj.name,
        "kind": "resource",
        "encoding": V3_RESOURCE_ENCODING[obj.encoding],
        "offset": obj.offset,
        "bytes": obj.bytes,
    }


def _components_json(plan: V3Plan) -> dict[str, dict[str, object]]:
    """The components section, with each component's configuration beside its resources.

    A draft package component states its ``target`` and its ``config`` at the
    component's own level: the engine reads ``component["target"]`` and
    ``component["config"]`` directly, so the plan's entries are spread onto
    the component rather than nested under a single ``config`` key.
    """
    components: dict[str, dict[str, object]] = {
        name: {"resources": dict(resources)}
        for name, resources in plan.components.items()
    }
    for name, entries in plan.component_configs.items():
        component = components.setdefault(name, {"resources": {}})
        component.update(entries)
    return components


def encode_v3_directory(
    identity: ArtifactIdentity,
    objects: Sequence[ArtifactObject],
    plan: V3Plan,
    payload_bytes: int,
    fingerprint: bytes,
) -> bytes:
    """Encode the v3 directory JSON."""

    if len(fingerprint) != 16:
        raise ArtifactError("container identity must be sixteen bytes")
    bindings: dict[str, object] = {}
    for binding in plan.bindings:
        bindings[binding.name] = binding.to_json()
    value = {
        "metadata": {
            "name": identity.model_id,
            "weights_id": identity.weights_id,
        },
        "files": [{"path": None, "payload_bytes": payload_bytes}],
        "objects": [_object_json(obj) for obj in objects],
        "bindings": bindings,
        "uses": [use.to_json() for use in plan.uses],
        "components": _components_json(plan),
    }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class V3ArtifactWriter:
    """Write one planned artifact payload at a time in directory order."""

    def __init__(
        self,
        path: str | Path,
        identity: ArtifactIdentity,
        specs: Sequence[ObjectSpec],
        plan: V3Plan,
        fingerprint: bytes = QUASAR_ID,
    ) -> None:
        self.path = Path(path)
        self.identity = identity
        self.plan = plan
        self.objects = plan_objects(specs)
        payload_bytes = self.objects[-1].offset + self.objects[-1].bytes
        directory = encode_v3_directory(
            identity, self.objects, plan, payload_bytes, fingerprint
        )
        self.payload_offset = align_up(V3_HEADER_BYTES + len(directory), PAYLOAD_ALIGNMENT)
        self._file = self.path.open("wb")
        self._file.write(struct.pack("<8sQ", MAGIC_V3, len(directory)))
        self._file.write(fingerprint)
        self._file.write(directory)
        self._file.write(b"\x00" * (self.payload_offset - V3_HEADER_BYTES - len(directory)))
        self._next = 0
        self._cursor = 0
        self._finished = False

    def write(self, name: str, payload: Payload) -> None:
        if self._finished:
            raise RuntimeError("artifact writer is already finished")
        if self._next >= len(self.objects):
            raise ArtifactError("artifact already has every planned payload")
        obj = self.objects[self._next]
        if name != obj.name:
            raise ArtifactError(f"expected payload {obj.name}, got {name}")
        if obj.offset > self._cursor:
            self._file.write(b"\x00" * (obj.offset - self._cursor))
            self._cursor = obj.offset
        written = 0
        for chunk in _payload_chunks(payload):
            if written + len(chunk) > obj.bytes:
                raise ArtifactError(f"payload {name} exceeds its planned byte length")
            self._file.write(chunk)
            written += len(chunk)
        if written != obj.bytes:
            raise ArtifactError(f"payload {name} has {written} bytes; expected {obj.bytes}")
        self._cursor = obj.offset + obj.bytes
        self._next += 1

    def finish(self) -> None:
        if self._finished:
            return
        if self._next != len(self.objects):
            missing = self.objects[self._next].name
            raise ArtifactError(f"artifact is missing payload {missing}")
        self._file.truncate(self.payload_offset + self._cursor)
        self._file.flush()
        self._file.close()
        self._finished = True

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "V3ArtifactWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            try:
                self.finish()
            finally:
                if not self._finished:
                    self.close()
        else:
            self.close()


__all__ = [
    "MAGIC_V3",
    "NEUTRAL_ID",
    "PAYLOAD_ALIGNMENT",
    "QUASAR_ID",
    "V3ArtifactWriter",
    "V3Binding",
    "V3Plan",
    "V3Use",
    "V3_HEADER_BYTES",
    "encode_v3_directory",
]
