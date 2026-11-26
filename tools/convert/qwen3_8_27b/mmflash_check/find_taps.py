"""Find which target layers the engine's captured features correspond to.

The engine captures the target's hidden states at five layers and feeds them
to the draft's feature projection.  The reference implementation takes the
hidden states of ``hidden_states[layer_id + 1]``, which is the output of
decoder layer ``layer_id`` counted from zero.  This checks the engine's
capture against both readings by projecting the engine's features and
comparing them with the projection of the same layers taken from a reference
run.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/chenjx/ninfer-mmflash")

from dflash.model import DFlashDraftModel  # noqa: E402

TRACE = Path("/data/models/mmflash-work/trace")
CHECKPOINT = "/data/models/FSCCS-Qwen3.8-27B-MMFlash"


def read_dump(name: str):
    path = TRACE.with_name(TRACE.name + "." + name + ".bin")
    raw = path.read_bytes()
    elements = struct.unpack_from("<q", raw, 0)[0]
    shape = struct.unpack_from("<4i", raw, 8)
    array = np.frombuffer(raw[28:], dtype=np.uint16)[:elements].copy()
    return array, (shape[0], shape[1])


def as_bf16(array: np.ndarray) -> torch.Tensor:
    return torch.frombuffer(np.ascontiguousarray(array).tobytes(),
                            dtype=torch.bfloat16).clone()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.to(torch.float32).flatten().double()
    b32 = b.to(torch.float32).flatten().double()
    return float(torch.nn.functional.cosine_similarity(a32, b32, dim=0))


def main() -> int:
    features, feat_shape = read_dump("context_features")
    projected, proj_shape = read_dump("context_projected")
    engine_features = as_bf16(features).view(feat_shape[0], feat_shape[1])
    engine_context = as_bf16(projected).view(proj_shape[0], proj_shape[1]).T.contiguous()

    draft = DFlashDraftModel.from_pretrained(CHECKPOINT, torch_dtype=torch.bfloat16)
    draft.eval()

    # The engine's capture is [feature_rows, tokens] with the five taps
    # stacked: tap i occupies rows [i * hidden, (i + 1) * hidden).
    hidden = 5120
    taps = 5
    tokens = engine_features.shape[1]
    engine_taps = [
        engine_features[i * hidden:(i + 1) * hidden, :].T.contiguous()
        for i in range(taps)
    ]

    print("the engine's per-tap statistics:")
    for index, tap in enumerate(engine_taps):
        print(f"  tap {index}: absmax={float(tap.abs().max()):.4f} "
              f"std={float(tap.float().std()):.5f}")

    # The reference projects the five taps as one concatenated vector.
    per_token = engine_features.T.contiguous().view(1, tokens, -1)
    with torch.no_grad():
        reference_context = draft.hidden_norm(draft.fc(per_token))[0]
    print()
    print("the engine's context vs the reference on the engine's own features:")
    print("  cos =", cos(engine_context, reference_context))
    print("  the engine's absmax:", float(engine_context.abs().max()))
    print("  the reference's    :", float(reference_context.abs().max()))

    # Does the engine's context match the reference's projection of SOME
    # ordering of its own taps?  That would mean the taps are right but the
    # order is not.
    print()
    print("the engine's context vs the reference with the taps permuted:")
    import itertools
    best = (None, -2.0)
    for perm in itertools.permutations(range(taps)):
        stacked = torch.cat([engine_taps[i] for i in perm], dim=-1).view(1, tokens, -1)
        with torch.no_grad():
            candidate = draft.hidden_norm(draft.fc(stacked))[0]
        value = cos(engine_context, candidate)
        if value > best[1]:
            best = (perm, value)
    print("  the best permutation:", best[0], " cos =", best[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
