"""Independent PyTorch CPU oracle for one captured Inkling rnn_hidden proposal block."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    return (y * weight.float()).bfloat16()


def rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # x: [token, head, 128], NeoX cross-half rotation.
    inv = 1.0 / (1000000.0 ** (torch.arange(0, 128, 2).float() / 128))
    angles = positions.float()[:, None] * inv[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    a, b = x[..., :64].float(), x[..., 64:].float()
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1).bfloat16()


def mix(features: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    value = (features.float() * alpha.float()[None, :, None]).sum(dim=1)
    return (value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
            * weight.float()).bfloat16()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("debug", type=Path)
    parser.add_argument("mtp", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from tools.artifact.container import Artifact

    with args.debug.open("rb") as f:
        hidden_size, length, anchor = struct.unpack("<3i", f.read(12))
        native_drafts = list(struct.unpack("<7i", f.read(28)))
        target_predictions = list(struct.unpack("<8i", f.read(32)))
        if hidden_size != 2304:
            raise ValueError("unexpected debug hidden size")
        words = torch.frombuffer(f.read(), dtype=torch.uint16)
        features = words.view(torch.bfloat16).reshape(4, length, hidden_size).permute(1, 0, 2).contiguous()
    print("native_anchor", anchor)
    print("native_drafts", native_drafts)
    print("target_predictions", target_predictions)

    weights = {}
    with Artifact(args.mtp) as artifact:
        for obj in artifact.objects:
            raw = artifact.payload(obj)
            weights[obj.name] = torch.frombuffer(raw, dtype=torch.uint16).view(
                torch.bfloat16).reshape(obj.shape).clone()
            del raw
        with safe_open(str(args.target), framework="pt", device="cpu") as source:
            lm_head = source.get_tensor("model.llm.unembed.weight")
            embed_norm = source.get_tensor("model.llm.embed_norm.weight")
            anchor_embedding = source.get_slice("model.llm.embed.weight")[anchor]
        W = lambda name: weights[name]
        alpha = torch.softmax(W("fuse_weight").float(), dim=-1).bfloat16()
        f_rnn = mix(features[-1:], alpha[3], W("markov_head.dflare_norm.weight"))[0]
        context_k, context_v = [], []
        positions = torch.arange(length + 8)
        for layer in range(3):
            prefix = f"layers.{layer}."
            mixed = mix(features, alpha[layer], W(prefix + "dflare_norm.weight"))
            key = F.linear(mixed, W(prefix + "self_attn.k_proj_ctx.weight"))
            value = F.linear(mixed, W(prefix + "self_attn.v_proj_ctx.weight"))
            key = rms(key.reshape(length, 8, 128), W(prefix + "self_attn.k_norm_ctx.weight"))
            key = rope(key, positions[:length])
            context_k.append(key)
            context_v.append(value.reshape(length, 8, 128))

        hidden = W("mask_embed")[None, :].expand(8, hidden_size).clone()
        hidden[0] = rms(anchor_embedding, embed_norm)
        block_positions = positions[length:length + 8]
        for layer in range(3):
            prefix = f"layers.{layer}."
            normed = rms(hidden, W(prefix + "input_layernorm.weight"))
            q = F.linear(normed, W(prefix + "self_attn.q_proj.weight")).reshape(8, 24, 128)
            k = F.linear(normed, W(prefix + "self_attn.k_proj.weight")).reshape(8, 8, 128)
            v = F.linear(normed, W(prefix + "self_attn.v_proj.weight")).reshape(8, 8, 128)
            q = rope(rms(q, W(prefix + "self_attn.q_norm.weight")), block_positions)
            k = rope(rms(k, W(prefix + "self_attn.k_norm.weight")), block_positions)
            all_k = torch.cat((context_k[layer], k), dim=0)
            all_v = torch.cat((context_v[layer], v), dim=0)
            attention = torch.empty_like(q)
            for token in range(8):
                keys = all_k[:length + token + 1]
                values = all_v[:length + token + 1]
                if layer in (0, 2):
                    start = max(0, length - 1024)
                    keys = all_k[start:length + token + 1]
                    values = all_v[start:length + token + 1]
                for head in range(24):
                    kv_head = head // 3
                    head_scores = (keys[:, kv_head].float() @ q[token, head].float()) * (128 ** -0.5)
                    p = torch.softmax(head_scores, dim=-1)
                    attention[token, head] = (p[:, None] * values[:, kv_head].float()).sum(dim=0).bfloat16()
            projected = F.linear(attention.reshape(8, 3072), W(prefix + "self_attn.o_proj.weight"))
            hidden = hidden + projected
            normed = rms(hidden, W(prefix + "post_attention_layernorm.weight"))
            gate = F.linear(normed, W(prefix + "mlp.gate_proj.weight"))
            up = F.linear(normed, W(prefix + "mlp.up_proj.weight"))
            projected = F.linear(F.silu(gate) * up, W(prefix + "mlp.down_proj.weight"))
            hidden = hidden + projected
        hidden = rms(hidden, W("norm.weight"))
        base_logits = F.linear(hidden[1:], lm_head)
        state = torch.zeros(256, dtype=torch.bfloat16)
        previous = anchor
        oracle_drafts = []
        for step in range(7):
            markov = W("markov_head.markov_w1.weight")[previous]
            inputs = torch.cat((state, markov, hidden[step + 1], f_rnn))
            projection = F.linear(inputs, W("markov_head.rnn_proj.weight"), W("markov_head.rnn_proj.bias"))
            gate = torch.sigmoid(projection[:256].float()).bfloat16().float()
            kept = (gate * state.float()).bfloat16().float()
            candidate = torch.tanh(projection[256:512].float()).bfloat16().float()
            blend = ((1 - gate).bfloat16().float() * candidate).bfloat16().float()
            state = (kept + blend).bfloat16()
            latent = torch.tanh(projection[512:].float()).bfloat16()
            # Unfolded equivalent of lm_head @ hidden_out; avoids a 100M-element temporary fold.
            markov_hidden = F.linear(latent, W("markov_head.hidden_out.weight"), W("markov_head.hidden_out.bias"))
            bias = F.linear(markov_hidden, lm_head)
            logits = (base_logits[step].float() + bias.float())[:200058]
            logits = (30 * torch.tanh(logits / 30)).bfloat16()
            previous = int(logits.argmax())
            oracle_drafts.append(previous)
        print("oracle_drafts", oracle_drafts)
        print("matching_positions", [a == b for a, b in zip(native_drafts, oracle_drafts)])


if __name__ == "__main__":
    main()
