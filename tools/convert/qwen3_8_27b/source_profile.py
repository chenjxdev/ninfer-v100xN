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
    #: How the checkpoint stores its weights. ``modelopt`` means every Linear
    #: is a ModelOpt quad (packed weight + block scales + per-tensor
    #: multiplier + input scale); ``bf16`` means plain BF16 matrices that the
    #: converter quantizes to NVFP4 itself (per-16-element e4m3 block scales,
    #: per-tensor multiplier ``amax/448``).
    source_encoding: str = "modelopt"
    #: The container the artifact is written in: ``v2`` (a flat object list)
    #: or ``v3`` (objects + bindings + uses + components).
    container_version: str = "v2"
    #: The weights identity the engine maps to a weights profile. The engine
    #: registers the full-NVFP4 (all-NVFP4 projections, W8 endpoints) format
    #: map as ``quasar-nvfp4``; ``nvfp4`` is the mixed NVFP4/FP8 map.
    weights_id: str = "nvfp4"
    #: The GDN layers the ARTIFACT stores as row-scaled FP8. None means the
    #: production format map: ten layers. The publisher's own FP8 export
    #: (``fp8_gdn_layers``) is a different thing -- it says where the
    #: checkpoint shipped FP8, not where the artifact keeps FP8.
    artifact_fp8_gdn_layers: frozenset[int] | None = None
    #: Whether every counted projection is an NVFP4 object (the full-NVFP4
    #: format map the engine registers as ``quasar-nvfp4``).
    full_nvfp4: bool = False

    @property
    def mtp_auxiliary_fields(self) -> tuple[str, ...]:
        if self.source_encoding == "bf16":
            # A BF16 checkpoint ships no MTP tensors at all, so the auxiliary
            # checkpoint provides the complete draft layer.
            return (
                "mtp.fc.weight",
                "mtp.layers.0.input_layernorm.weight",
                "mtp.layers.0.mlp.down_proj.weight",
                "mtp.layers.0.mlp.gate_proj.weight",
                "mtp.layers.0.mlp.up_proj.weight",
                "mtp.layers.0.post_attention_layernorm.weight",
                "mtp.layers.0.self_attn.k_norm.weight",
                "mtp.layers.0.self_attn.k_proj.weight",
                "mtp.layers.0.self_attn.o_proj.weight",
                "mtp.layers.0.self_attn.q_norm.weight",
                "mtp.layers.0.self_attn.q_proj.weight",
                "mtp.layers.0.self_attn.v_proj.weight",
                "mtp.norm.weight",
                "mtp.pre_fc_norm_embedding.weight",
                "mtp.pre_fc_norm_hidden.weight",
            )
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

# kelsey1337/DaisyRose-27B-Heretic-v2 is the ablation of RadixArk/DaisyRose-27B
# shipped as plain BF16: no ModelOpt quads at all, so every projection is
# quantized to NVFP4 by the converter itself and every one of the 256 counted
# projections is an NVFP4 object (the full-NVFP4 grade the engine recognizes).
# The checkpoint carries no MTP tensors -- the ablation tool dropped the draft
# layer on save -- so the auxiliary checkpoint supplies all fifteen of them.

# The same artifact as the QUASAR weights profile: every one of the 256
# counted projections is an NVFP4 object and both vocabulary endpoints are
# W8, which is the format map the engine's QUASAR adapter binds and the only
# one that unlocks --volta-fast-decode. The engine registers that profile by
# the container identity (the sixteen bytes at offset 16 of a v3 container),
# so the artifact is written in the v3 container.
HERETIC_QUASAR = SourceProfile(
    name="heretic-quasar",
    repository="kelsey1337/DaisyRose-27B-Heretic-v2",
    revision="78cc5e82ee867e22217f02ae30f1d26e8b953f33",
    fp8_gdn_layers=frozenset(),
    fp8_full_attention=False,
    mtp_auxiliary=None,
    embedding_format="W8G32_F16S",
    output_head_format="W8G32_F16S",
    source_encoding="bf16",
    container_version="v3",
    weights_id="quasar-nvfp4",
    artifact_fp8_gdn_layers=frozenset(),
    full_nvfp4=True,
)

# The production format map over a BF16 source: the attention projections and
# ten GDN layers are row-scaled FP8, every other projection is NVFP4, both
# vocabulary endpoints are Q6 and the MTP matrices are Q4 -- exactly the map
# of the production artifact, so the engine binds it the same way (the mixed
# ``nvfp4`` weights profile) and the same start-up flags apply.
HERETIC_PLAIN = SourceProfile(
    name="heretic-plain",
    repository="kelsey1337/DaisyRose-27B-Heretic-v2",
    revision="78cc5e82ee867e22217f02ae30f1d26e8b953f33",
    fp8_gdn_layers=frozenset((0, 1, 2, 4, 5, 57, 58, 60, 61, 62)),
    fp8_full_attention=True,
    mtp_auxiliary=None,
    embedding_format="Q6G64_F16S",
    output_head_format="Q6G64_F16S",
    source_encoding="bf16",
)

# The same artifact with W8 endpoints: the engine's full-NVFP4 grade requires
# both vocabulary endpoints in q8_g32_fp16, and the BF16 source carries no
# redundancy at 8 bit, so W8 is a free widening rather than wasted storage.

PROFILES: dict[str, SourceProfile] = {
    QAD.name: QAD,
    DAISYROSE.name: DAISYROSE,
    HERETIC_QUASAR.name: HERETIC_QUASAR,
    HERETIC_PLAIN.name: HERETIC_PLAIN,
}

_ACTIVE_NAME = os.environ.get("NINFER_NVFP4_SOURCE", QAD.name)
if _ACTIVE_NAME not in PROFILES:
    raise ValueError(
        f"unknown NINFER_NVFP4_SOURCE {_ACTIVE_NAME!r}; "
        f"expected one of {sorted(PROFILES)}"
    )
ACTIVE: SourceProfile = PROFILES[_ACTIVE_NAME]
