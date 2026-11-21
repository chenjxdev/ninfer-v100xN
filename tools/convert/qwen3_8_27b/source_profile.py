"""Closed source profiles for the Qwen3.8-27B mixed NVFP4/FP8 conversion.

A profile binds one quantized checkpoint to the recipe's expectations: the
repository and revision it came from, which layers' GDN quads the publisher
exported as ModelOpt FP8 (per-tensor) rather than NVFP4 (block-scaled), and
whether the checkpoint ships the four MTP tensors that the NVIDIA QAD export
omits.

The profile is selected before the recipe is imported::

    NINFER_NVFP4_SOURCE=daisyrose python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \\
        --model /data/models/daisyrose-27b-nvfp4 \\
        --out /data/models/daisyrose_27b_nvfp4.ninfer

Two checkpoints are supported:

``qad``
    NVIDIA's ``nvidia/Qwen3.8-27B-NVFP4-QAD`` at ``modelopt-0.46.0rc1``. Its
    export is MIXED_PRECISION: every full-attention layer's quads and ten GDN
    layers' quads are ModelOpt FP8 per-tensor quads, the rest are NVFP4
    block-scaled quads. It ships only the eleven MTP layer tensors, so
    ``mtp.fc``, the two pre-fc norms and ``mtp.norm`` are read from the
    auxiliary Huihui checkpoint.

``daisyrose``
    ``RadixArk/DaisyRose-27B-NVFP4`` at the pinned revision. The same
    MIXED_PRECISION export, but every GDN layer is FP8 in the source (208 FP8
    quads against QAD's 94), and the checkpoint ships all fifteen MTP tensors,
    so no auxiliary checkpoint is needed. The artifact keeps the production
    profile's ten FP8 GDN layers and requantizes the remaining thirty-eight
    from FP8 to NVFP4 (the same precision the production QAD artifact stores
    there), which keeps the artifact's format map -- and therefore its size --
    identical to the production artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """One quantized checkpoint's closed expectations."""

    name: str
    repository: str
    revision: str
    #: GDN layers whose quads the publisher exported as ModelOpt FP8.
    fp8_gdn_layers: frozenset[int]
    #: Whether every full-attention layer's quads are ModelOpt FP8.
    fp8_full_attention: bool
    #: Checkpoint providing the four MTP tensors QAD omits; None when the
    #: source ships them itself.
    mtp_auxiliary: str | None
    #: Storage of the two vocabulary endpoints.
    #:
    #: The input embedding is a per-device replica and the output head is
    #: split across the tensor-parallel devices, so the endpoint choice is
    #: also a VRAM decision. A container wider than its source carries no
    #: information: the QAD head is NVFP4 (4 bit) and the DaisyRose head is
    #: NVFP4 too, so an 8-bit head stores four bits of redundancy per weight.
    embedding_format: str = "Q6G64_F16S"
    output_head_format: str = "W8G32_F16S"

    @property
    def mtp_auxiliary_fields(self) -> tuple[str, ...]:
        return (
            "mtp.fc.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.norm.weight",
        )


QAD = SourceProfile(
    name="qad",
    repository="nvidia/Qwen3.8-27B-NVFP4-QAD",
    revision="modelopt-0.46.0rc1",
    fp8_gdn_layers=frozenset((0, 1, 2, 4, 5, 57, 58, 60, 61, 62)),
    fp8_full_attention=True,
    mtp_auxiliary=(
        "/data/models/KatterMobile/"
        "Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-NVFP4-QAT"
    ),
    embedding_format="Q6G64_F16S",
    output_head_format="W8G32_F16S",
)

DAISYROSE = SourceProfile(
    name="daisyrose",
    repository="RadixArk/DaisyRose-27B-NVFP4",
    revision="40bf42fdff1dc2d619ea01d564f1597363216ebb",
    fp8_gdn_layers=frozenset(range(64)) - frozenset(range(3, 64, 4)),
    fp8_full_attention=True,
    mtp_auxiliary=None,
    # The embedding is a per-device replica of a BF16 source: Q6 is a real
    # down-quantization, not redundant storage. The head is NVFP4 in the
    # checkpoint, so Q6 is already wider than its information.
    embedding_format="Q6G64_F16S",
    output_head_format="Q6G64_F16S",
)

PROFILES: dict[str, SourceProfile] = {
    QAD.name: QAD,
    DAISYROSE.name: DAISYROSE,
}

_ACTIVE_NAME = os.environ.get("NINFER_NVFP4_SOURCE", QAD.name)
if _ACTIVE_NAME not in PROFILES:
    raise ValueError(
        f"unknown NINFER_NVFP4_SOURCE {_ACTIVE_NAME!r}; "
        f"expected one of {sorted(PROFILES)}"
    )
ACTIVE: SourceProfile = PROFILES[_ACTIVE_NAME]
