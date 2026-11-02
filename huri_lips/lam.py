"""The network of LAM Audio2Expression (Alibaba 3DAIGC, Apache-2.0,
https://github.com/aigc3d/LAM_Audio2Expression), written again for inference
only so that its published weights (3DAIGC/LAM_audio2exp on Hugging Face,
"streaming") load as they are."""

import os
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from huri_lips.config import (
    CHECKPOINT,
    HF_FILE,
    HF_REPO,
    HF_REVISION,
    IDENTITIES,
    SHAPES,
    WINDOW_FRAMES,
    models_dir,
)

# facebook/wav2vec2-base-960h's architecture (LAM's configs/wav2vec2_config.json);
# its weights come with LAM's checkpoint, nothing is downloaded from it.
_WAV2VEC2 = dict(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    conv_dim=[512] * 7,
    conv_kernel=[10, 3, 3, 3, 3, 2, 2],
    conv_stride=[5, 2, 2, 2, 2, 2, 2],
    conv_bias=False,
    feat_extract_norm="group",
    feat_extract_activation="gelu",
    num_conv_pos_embeddings=128,
    num_conv_pos_embedding_groups=16,
    do_stable_layer_norm=False,
    layer_norm_eps=1e-5,
    apply_spec_augment=True,
    mask_time_prob=0.05,
    vocab_size=32,
)


class _ConvNormRelu(nn.Module):
    def __init__(self, c_in: int, c_out: int, residual: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(c_out)
        self.residual = residual
        if residual and c_in != c_out:
            self.residual_layer = nn.Sequential(nn.Conv1d(c_in, c_out, 3, 1, 1))
        else:
            self.residual_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(self.conv(x).transpose(1, 2)).transpose(1, 2)
        if self.residual:
            out = out + (x if self.residual_layer is None else self.residual_layer(x))
        return F.relu(out)


class _SeqTranslator(nn.Module):
    def __init__(self, c_in: int, c_out: int, layers: int):
        super().__init__()
        blocks = [_ConvNormRelu(c_in, c_out, residual=True)]
        blocks += [
            _ConvNormRelu(c_out, c_out, residual=True) for _ in range(layers - 1)
        ]
        self.conv_layers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class _IdentityEncoder(nn.Module):
    def __init__(self, hidden: int, identities: int, identity_dim: int = 64):
        super().__init__()
        self.id_mlp = nn.Conv1d(identities, identity_dim, 1, 1)
        self.first_net = _SeqTranslator(hidden + identity_dim, hidden, layers=3)

    def forward(self, features: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        ident = identity.to(features.dtype).unsqueeze(-1)
        ident = self.id_mlp(ident.expand(-1, -1, features.shape[2]))
        return self.first_net(torch.cat([features, ident], dim=1))


class LamA2E(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        config = Wav2Vec2Config(**_WAV2VEC2)
        config._attn_implementation = "eager"  # exports to ONNX as it runs
        self.audio_encoder = Wav2Vec2Model(config)
        self.feature_projection = nn.Linear(config.hidden_size, hidden)
        self.identity_encoder = _IdentityEncoder(hidden, IDENTITIES)
        self.decoder = nn.ModuleList(
            [nn.Sequential(*[_ConvNormRelu(hidden, hidden) for _ in range(3)])]
        )
        self.output_proj = nn.Linear(hidden, SHAPES)

    def forward(
        self, audio: torch.Tensor, identity: torch.Tensor, frames: int = WINDOW_FRAMES
    ) -> torch.Tensor:
        enc = self.audio_encoder
        hidden = enc.feature_extractor(audio).transpose(1, 2)
        hidden = F.interpolate(
            hidden.transpose(1, 2), size=frames, mode="linear", align_corners=True
        ).transpose(1, 2)
        hidden = enc.feature_projection(hidden)[0]
        hidden = enc.encoder(hidden).last_hidden_state
        x = self.feature_projection(hidden).transpose(1, 2)
        x = self.identity_encoder(x, identity)
        x = self.decoder[0](x).transpose(1, 2)
        return torch.sigmoid(self.output_proj(x))


def one_hot_identity(identity: int, batch: int = 1) -> torch.Tensor:
    if not 0 <= identity < IDENTITIES:
        raise ValueError(f"identity must be in 0..{IDENTITIES - 1}")
    return F.one_hot(torch.full((batch,), identity), IDENTITIES).float()


# --- weights -----------------------------------------------------------------

_UNUSED = (
    "audio_encoder.lm_head.",
    "identity_encoder.grus.",
)


def _rename(state: Dict[str, torch.Tensor], model: LamA2E) -> Dict[str, torch.Tensor]:
    own = model.state_dict()
    out = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("backbone.")
        if key.startswith(_UNUSED):
            continue
        if key not in own:
            # Recent torch keeps the positional conv's weight norm as a
            # parametrization: weight_g/weight_v are original0/original1.
            alt = key.replace("conv.weight_g", "conv.parametrizations.weight.original0")
            alt = alt.replace("conv.weight_v", "conv.parametrizations.weight.original1")
            key = alt
        out[key] = value
    return out


def _load_into(model: LamA2E, state: Dict[str, torch.Tensor]) -> LamA2E:
    state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(_rename(state, model), strict=False)
    # masked_spec_embed only serves training-time masking.
    missing = [k for k in missing if k != "audio_encoder.masked_spec_embed"]
    if missing or unexpected:
        raise RuntimeError(
            f"LAM weights do not fit: missing {missing[:5]}, "
            f"unexpected {unexpected[:5]}"
        )
    return model


def load_model(checkpoint: Optional[os.PathLike] = None, device: str = "cpu") -> LamA2E:
    path = Path(checkpoint) if checkpoint else fetch_checkpoint()
    state = torch.load(path, map_location="cpu", weights_only=True)
    return _load_into(LamA2E(), state).eval().to(device)


def fetch_checkpoint(dest: Optional[Path] = None) -> Path:
    dest = dest or models_dir()
    path = dest / CHECKPOINT
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        archive = hf_hub_download(HF_REPO, HF_FILE, revision=HF_REVISION, local_dir=tmp)
        return import_checkpoint(Path(archive), dest)


def import_checkpoint(source: Path, dest: Optional[Path] = None) -> Path:
    dest = dest or models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / CHECKPOINT
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        raw = source
        if tarfile.is_tarfile(source):
            with tarfile.open(source) as tar:
                member = next(
                    (m for m in tar.getmembers() if m.name.endswith("_streaming.tar")),
                    None,
                )
                stream = tar.extractfile(member) if member else None
                if stream is None:
                    raise RuntimeError(f"{source}: no streaming checkpoint inside")
                raw = Path(tmp) / "checkpoint.zip"
                raw.write_bytes(stream.read())
        state = torch.load(raw, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {
        key: value
        for key, value in state.items()
        if not key.removeprefix("module.").removeprefix("backbone.").startswith(_UNUSED)
    }
    _load_into(LamA2E(), state)  # refuse weights that do not fit, now
    partial = path.with_suffix(".part")
    torch.save(state, partial)
    partial.replace(path)
    return path


class _Exported(nn.Module):
    def __init__(self, model: LamA2E):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.model(audio, identity, WINDOW_FRAMES)


def export_onnx(out: Path, checkpoint: Optional[os.PathLike] = None) -> Path:
    from huri_lips.config import WINDOW_SAMPLES

    model = load_model(checkpoint, "cpu")
    out.parent.mkdir(parents=True, exist_ok=True)
    audio = torch.zeros(2, WINDOW_SAMPLES)
    identity = one_hot_identity(0, 2)
    partial = out.with_name(out.name + ".part")
    with torch.inference_mode():
        torch.onnx.export(
            _Exported(model),
            (audio, identity),
            str(partial),
            input_names=["audio", "identity"],
            output_names=["weights"],
            dynamic_axes={
                "audio": {0: "batch"},
                "identity": {0: "batch"},
                "weights": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    partial.replace(out)
    return out
