"""Compare the engine's first-generation draft with the reference, step by step.

The engine dumps the captured target features, the context projection, the
draft's hidden state and its proposed tokens.  This feeds the engine's own
captured features into the reference implementation and compares each stage,
so a mismatch names the stage that went wrong.
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
DTYPE_BF16 = 0
DTYPE_I32 = 2


def read_dump(name: str):
    path = TRACE.with_name(TRACE.name + "." + name + ".bin")
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    elements = struct.unpack_from("<q", raw, 0)[0]
    shape = struct.unpack_from("<4i", raw, 8)
    dtype = struct.unpack_from("<I", raw, 24)[0]
    body = raw[28:]
    if dtype == DTYPE_I32:
        array = np.frombuffer(body, dtype=np.int32)[:elements].copy()
    else:
        array = np.frombuffer(body, dtype=np.uint16)[:elements].copy()
    return array, (shape[0], shape[1])


def as_bf16(array: np.ndarray) -> torch.Tensor:
    return torch.frombuffer(np.ascontiguousarray(array).tobytes(),
                            dtype=torch.bfloat16).clone()


def report(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    diff = (a32 - b32).abs()
    scale = b32.abs().max().clamp(min=1e-9)
    cos = torch.nn.functional.cosine_similarity(
        a32.flatten().double(), b32.flatten().double(), dim=0)
    print(f"  {label:<38} max_diff={float(diff.max()):.5f} "
          f"rel={float(diff.max() / scale):.5f} cos={float(cos):.7f}")


def main() -> int:
    features, feat_shape = read_dump("context_features")
    projected, proj_shape = read_dump("context_projected")
    hidden, hidden_shape = read_dump("draft_hidden")
    tokens, token_shape = read_dump("draft_tokens")
    embedding, emb_shape = read_dump("block_embedding")
    positions, pos_shape = read_dump("proposal_positions")

    draft = DFlashDraftModel.from_pretrained(CHECKPOINT, torch_dtype=torch.bfloat16)
    draft.eval()

    # The engine stores the captured features column-major: [feature_rows, tokens].
    engine_features = as_bf16(features).view(feat_shape[0], feat_shape[1])
    ctx_len = feat_shape[1]
    per_token = engine_features.T.contiguous().view(1, ctx_len, -1)

    # ---- stage 1: the context projection ----
    with torch.no_grad():
        reference_context = draft.hidden_norm(draft.fc(per_token))[0]
    engine_context = as_bf16(projected).view(proj_shape[0], proj_shape[1]).T.contiguous()

    print("stage 1, the context projection hidden_norm(fc(features)):")
    print("  the engine's context:", tuple(engine_context.shape))
    print("  the reference's     :", tuple(reference_context.shape))
    if engine_context.shape == reference_context.shape:
        report("context projection", engine_context, reference_context)

    # ---- stage 2: the draft's own forward ----
    noise = as_bf16(embedding).view(emb_shape[0], emb_shape[1]).T.contiguous()
    block_len = noise.shape[0]
    block_positions = torch.as_tensor(positions, dtype=torch.int64).view(-1)
    # The reference rotates the context keys with the same table as the block
    # queries, so the positions cover the context and the block together.
    context_positions = block_positions[:1] - ctx_len + torch.arange(ctx_len)
    position_ids = torch.cat([context_positions, block_positions], dim=0).view(1, -1)

    with torch.no_grad():
        reference_hidden = draft(
            position_ids=position_ids,
            noise_embedding=noise.view(1, block_len, -1),
            target_hidden=per_token,
            use_cache=False,
        )[0]

    engine_hidden = as_bf16(hidden).view(hidden_shape[0], hidden_shape[1]).T.contiguous()
    print()
    print("stage 2, the draft forward:")
    print("  the engine's hidden:", tuple(engine_hidden.shape))
    print("  the reference's    :", tuple(reference_hidden.shape))
    k = min(engine_hidden.shape[0], reference_hidden.shape[0])
    report("draft hidden (engine vs reference)", engine_hidden[:k], reference_hidden[-k:])

    print()
    print("the engine's proposed tokens:", tokens[:8].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
