import os
import json
import argparse
import subprocess
import datetime
from typing import Optional, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from src.shared.data.io import load_param_split, flatten_param_folders
from src.shared.data.transport_conditioning import (
    load_transport_metadata,
    make_condition_maps,
    param_folder_from_path,
)
from src.obj1.conference.train_pix2pix_multioutput_patch_logc import Pix2PixGenerator


# =========================================================
# Shared helpers
# =========================================================
def load_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["stats"] if isinstance(j, dict) and "stats" in j else j


def normalize_K(K: np.ndarray, stats: dict, use_logK: bool, eps_k: float) -> np.ndarray:
    K = K.astype(np.float64)
    if use_logK:
        K = np.log(np.clip(K, eps_k, None))
    K = (K - float(stats["k_mean"])) / (float(stats["k_std"]) + 1e-8)
    return K.astype(np.float32)


def gt_log10(C_phys: np.ndarray, eps_c: float) -> np.ndarray:
    return np.log10(np.clip(C_phys, 0.0, None) + eps_c).astype(np.float32)


def log10_to_phys(C_log10: np.ndarray, eps_c: float) -> np.ndarray:
    return np.clip((10.0 ** C_log10) - eps_c, 0.0, None).astype(np.float32)


def hann2d(patch: int) -> np.ndarray:
    w = np.hanning(patch).astype(np.float32)
    w2 = np.outer(w, w).astype(np.float32)
    return np.clip(w2, 1e-3, 1.0)


def resolve_ssim_data_range(args: argparse.Namespace) -> float:
    """
    Resolve a fixed SSIM data_range to ensure comparability across samples/timesteps.

    Recommended usage: evaluate SSIM on a fixed log-concentration range shared by all
    samples and all timesteps. By default this script uses [ssim_log_min, ssim_log_max],
    which defaults to [-12, 2] => data_range = 14. Override these bounds if your
    preprocessing uses a different target range.
    """
    if args.ssim_data_range is not None:
        dr = float(args.ssim_data_range)
    else:
        dr = float(args.ssim_log_max) - float(args.ssim_log_min)
    if dr <= 0.0:
        raise ValueError(f"SSIM data_range must be positive, got {dr}")
    return dr


def ssim_fixed(gt: np.ndarray, pr: np.ndarray, data_range: float) -> float:
    return float(ssim(gt, pr, data_range=float(data_range)))


def bbox_from_mask(mask: np.ndarray, pad: int = 8) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    return y0, y1 + 1, x0, x1 + 1


# =========================================================
# Time-conditioning helper
# =========================================================
def make_time_plane(t_idx: int, T: int, H: int, W: int, mode: str = "linear_0_1") -> np.ndarray:
    """
    Default assumption for time-conditioned model:
    use a constant plane with normalized timestep in [0,1] or [-1,1].
    """
    if T <= 1:
        val = 0.0
    else:
        val = float(t_idx) / float(T - 1)

    if mode == "linear_-1_1":
        val = 2.0 * val - 1.0

    return np.full((H, W), val, dtype=np.float32)


# =========================================================
# Stitching
# =========================================================
@torch.no_grad()
def stitch_predict_allT_multioutput(
    model,
    K_norm: np.ndarray,
    patch: int,
    stride: int,
    device: str,
    extra_channels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    For models that predict all T timesteps at once.
    Input:  (H, W)
    Output: (T, H, W)
    """
    H, W = K_norm.shape
    if patch > H or patch > W:
        raise ValueError(f"patch={patch} is larger than field size {(H, W)}")

    pred_sum = None
    w_sum = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
    if len(ys) == 0:
        ys = [0]
    if len(xs) == 0:
        xs = [0]
    if ys[-1] != H - patch:
        ys.append(H - patch)
    if xs[-1] != W - patch:
        xs.append(W - patch)

    w_patch = hann2d(patch)

    for y0 in ys:
        for x0 in xs:
            Kp = K_norm[y0:y0 + patch, x0:x0 + patch]
            if extra_channels is None:
                X_np = Kp[None, ...]
            else:
                Cp = extra_channels[:, y0:y0 + patch, x0:x0 + patch]
                X_np = np.concatenate([Kp[None, ...], Cp], axis=0)
            X = torch.from_numpy(X_np[None, ...]).to(device)
            out = model(X).squeeze(0).detach().cpu().numpy().astype(np.float32)  # (T, P, P)

            if pred_sum is None:
                T = out.shape[0]
                pred_sum = np.zeros((T, H, W), dtype=np.float32)

            pred_sum[:, y0:y0 + patch, x0:x0 + patch] += out * w_patch[None, :, :]
            w_sum[y0:y0 + patch, x0:x0 + patch] += w_patch

    w_sum = np.maximum(w_sum, 1e-6)
    return pred_sum / w_sum[None, :, :]


@torch.no_grad()
def stitch_predict_oneT_timecond(
    model,
    K_norm: np.ndarray,
    t_idx: int,
    T: int,
    patch: int,
    stride: int,
    device: str,
    time_mode: str = "linear_0_1",
    extra_channels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    For time-conditioned model that predicts one timestep at a time.
    Input:  (2+n, H, W) = [K, optional condition channels..., time_plane]
    Output: (H, W)
    """
    H, W = K_norm.shape
    if patch > H or patch > W:
        raise ValueError(f"patch={patch} is larger than field size {(H, W)}")

    pred_sum = np.zeros((H, W), dtype=np.float32)
    w_sum = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
    if len(ys) == 0:
        ys = [0]
    if len(xs) == 0:
        xs = [0]
    if ys[-1] != H - patch:
        ys.append(H - patch)
    if xs[-1] != W - patch:
        xs.append(W - patch)

    w_patch = hann2d(patch)

    for y0 in ys:
        for x0 in xs:
            Kp = K_norm[y0:y0 + patch, x0:x0 + patch]
            tp = make_time_plane(t_idx, T, patch, patch, mode=time_mode)
            channels = [Kp]
            if extra_channels is not None:
                channels.extend(extra_channels[:, y0:y0 + patch, x0:x0 + patch])
            channels.append(tp)
            X = np.stack(channels, axis=0)[None, ...]  # (1, C, P, P)
            X = torch.from_numpy(X).to(device)
            out = model(X).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)  # (P, P)

            pred_sum[y0:y0 + patch, x0:x0 + patch] += out * w_patch
            w_sum[y0:y0 + patch, x0:x0 + patch] += w_patch

    w_sum = np.maximum(w_sum, 1e-6)
    return pred_sum / w_sum


@torch.no_grad()
def stitch_predict_allT_timecond(
    model,
    K_norm: np.ndarray,
    patch: int,
    stride: int,
    device: str,
    T: int = 25,
    time_mode: str = "linear_0_1",
    extra_channels: Optional[np.ndarray] = None,
) -> np.ndarray:
    preds = []
    for t in range(T):
        pr = stitch_predict_oneT_timecond(
            model=model,
            K_norm=K_norm,
            t_idx=t,
            T=T,
            patch=patch,
            stride=stride,
            device=device,
            time_mode=time_mode,
            extra_channels=extra_channels,
        )
        preds.append(pr)
    return np.stack(preds, axis=0).astype(np.float32)


# =========================================================
# Model loader
# =========================================================
class _TimeCOndMapOnly(torch.nn.Module):
    """
    Wrapper so stitch code can call model(X) and get a single tensor.
    UNet2D_TimeCond_MultiHead returns (pred_map, pred_mass).
    Eval stitching calls model(X).squeeze() which fails on a tuple.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        pred_map, _ = self.base(x)
        return pred_map


def _remap_cta_state_dict(sd):
    remapped = {}
    for key, value in sd.items():
        new_key = key
        new_key = new_key.replace("btn_conv.", "bottleneck_conv.")
        new_key = new_key.replace("btn_attn.", "bottleneck_attn.")
        new_key = new_key.replace("bottleneck_attn.ln1.", "bottleneck_attn.norm1.")
        new_key = new_key.replace("bottleneck_attn.ln2.", "bottleneck_attn.norm2.")
        new_key = new_key.replace("temporal_attn.ln1.", "temporal_attn.norm1.")
        new_key = new_key.replace("temporal_attn.ln2.", "temporal_attn.norm2.")
        remapped[new_key] = value
    return remapped


def build_model_from_ckpt(ckpt, device: str):
    cfg = ckpt.get("cfg", {})
    sd = ckpt.get("model")

    if (
        sd is not None
        and (
            cfg.get("model_family") == "cta_unet"
            or any(k.startswith("enc0.") for k in sd.keys())
        )
    ):
        from src.obj1.conference.temporal_attn_unet import CTAUNet

        model = CTAUNet(
            in_ch=int(cfg.get("input_channels", 1)),
            out_ch=25,
            base=int(cfg.get("base_channels", cfg.get("base", 64))),
            t_embed_dim=int(cfg.get("t_embed_dim", 25)),
            t_heads=int(cfg.get("t_heads", 5)),
            pool_stride=int(cfg.get("pool_stride", 4)),
        ).to(device)
        model.load_state_dict(_remap_cta_state_dict(sd), strict=True)
        model.eval()
        return model, "model", "multioutput"

    if (
        sd is not None
        and (
            cfg.get("model_family") == "swin_unet"
            or any(k.startswith("patch_embed.") for k in sd.keys())
        )
    ):
        from src.obj1.conference.swin_unet import SwinUNet

        model = SwinUNet(
            in_ch=1,
            out_ch=25,
            base_dim=int(cfg.get("base_dim", 96)),
            window_size=int(cfg.get("window_size", 8)),
        ).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "multioutput"

    if cfg.get("model_family") in {
        "cta_unet",
        "cta_unet_ffl_alphaL",
        "cta_unet_ffl_alphaT_ratio",
        "cta_unet_ffl_transport_conditioned",
    }:
        from src.obj1.conference.temporal_attn_unet import CTAUNet

        model = CTAUNet(
            in_ch=int(cfg.get("input_channels", 1)),
            out_ch=25,
            base=int(cfg.get("base_channels", cfg.get("base", 64))),
            t_embed_dim=int(cfg.get("t_embed_dim", 25)),
            t_heads=int(cfg.get("t_heads", 5)),
            pool_stride=int(cfg.get("pool_stride", 4)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model, "model", "multioutput"

    if cfg.get("model_family") == "swin_unet":
        from src.obj1.conference.swin_unet import SwinUNet

        model = SwinUNet(
            in_ch=1,
            out_ch=25,
            base_dim=int(cfg.get("base_dim", 96)),
            window_size=int(cfg.get("window_size", 8)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model, "model", "multioutput"

    if cfg.get("model_family") == "mixvision_unet":
        from src.obj1.conference.mixvision_unet import MixVisionUNet

        model = MixVisionUNet(
            in_ch=1,
            out_ch=25,
            base_dim=int(cfg.get("base_dim", 48)),
            decoder_dim=int(cfg.get("decoder_dim", 128)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model, "model", "multioutput"

    if cfg.get("model_family") in {"deeponet2d", "deeponet_transport_conditioned"}:
        from src.obj1.conference.deeponet2d import DeepONet2D

        model = DeepONet2D(
            in_ch=int(cfg.get("input_channels", 1)),
            out_ch=25,
            branch_width=int(cfg.get("branch_width", 128)),
            branch_latent=int(cfg.get("branch_latent", 384)),
            trunk_width=int(cfg.get("trunk_width", 384)),
            basis_rank=int(cfg.get("basis_rank", 96)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model, "model", "multioutput"

    if cfg.get("model_family") == "pmt_unet_aspp_attn":
        from src.obj1.conference.pmt_unet_aspp_attn import PMT_UNet_ASPP_Attn

        model = PMT_UNet_ASPP_Attn(
            in_ch=1,
            out_ch=25,
            base=int(cfg.get("base_channels", 64)),
            attn_heads=int(cfg.get("attn_heads", 4)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model, "model", "multioutput"

    if cfg.get("model_family") == "pgt_ufno":
        from src.obj1.conference.pgt_ufno import PGT_UFNO

        model = PGT_UFNO(
            in_ch=1,
            out_ch=25,
            width=int(cfg.get("width", 64)),
            modes1=int(cfg.get("modes1", 20)),
            modes2=int(cfg.get("modes2", 20)),
            depth=int(cfg.get("depth", 4)),
            local_base=int(cfg.get("local_base", 32)),
            temporal_hidden=int(cfg.get("temporal_hidden", 8)),
            use_coords=bool(cfg.get("use_coords", True)),
            pad_ratio=float(cfg.get("pad_ratio", 0.125)),
            plume_center=float(cfg.get("plume_center", -6.0)),
            plume_scale=float(cfg.get("plume_scale", 1.5)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model, "model", "multioutput"

    if cfg.get("model_family") in {"fno2d", "fno_transport_conditioned"}:
        from src.obj1.conference.fno2d import FNO2D

        model = FNO2D(
            in_ch=int(cfg.get("input_channels", 1)),
            out_ch=25,
            width=int(cfg.get("width", cfg.get("fno_width", 64))),
            modes1=int(cfg.get("modes1", cfg.get("fno_modes1", 20))),
            modes2=int(cfg.get("modes2", cfg.get("fno_modes2", 20))),
            depth=int(cfg.get("depth", cfg.get("fno_depth", 4))),
            use_coords=bool(cfg.get("use_coords", True)),
            pad_ratio=float(cfg.get("pad_ratio", 0.125)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model, "model", "multioutput"

    # ---------- pix2pix ----------
    if "gen" in ckpt:
        model = Pix2PixGenerator(in_ch=int(cfg.get("input_channels", 1)), out_ch=25).to(device)
        model.load_state_dict(ckpt["gen"], strict=True)
        model.eval()
        return model, "gen", "multioutput"

    # ---------- standard checkpoints ----------
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint does not contain 'model' or 'gen'. Keys={list(ckpt.keys())}")

    sd = ckpt["model"]

    if cfg.get("model_family") == "pix2pix_transport_conditioned":
        model = Pix2PixGenerator(in_ch=int(cfg.get("input_channels", 1)), out_ch=25).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "multioutput"

    # ---------- TimeCond UNet — detect by cfg BEFORE key scanning ----------
    # TimeCond uses 'inc.net.0.weight' naming (same as Option B), but its
    # forward() returns a tuple (pred_map, pred_mass). Detect early via cfg
    # fields that are exclusive to the TimeCond training script so we can
    # wrap the model before returning.
    is_timecond = (
        "lambda_mass_head" in cfg
        or "t_index_mode"  in cfg
        or "mass_mlp_hidden" in cfg
    )
    if is_timecond:
        from src.obj1.conference.unet2d_timecond_multihead import UNet2D_TimeCond_MultiHead
        base   = int(cfg.get("base_channels",   64))
        hidden = int(cfg.get("mass_mlp_hidden", 256))
        _inner = UNet2D_TimeCond_MultiHead(
            in_ch=int(cfg.get("input_channels", 2)), base=base, mass_mlp_hidden=hidden
        ).to(device)
        _inner.load_state_dict(sd, strict=True)
        _inner.eval()
        model = _TimeCOndMapOnly(_inner).to(device)
        model.eval()
        return model, "model", "timecond"

    # Scan for the TRUE first conv layer (the one that reads the raw input channels).
    # Priority order — first match wins:
    #   "inc.net.0.weight"   → UNet_ASPP_Attn (Option B) and UNet2D standard naming
    #   "enc1.net.0.weight"  → older UNet2D variant naming
    # REMOVED "d1.conv.net.0.weight": that is the SECOND conv stage (Down block),
    #   which has in_ch=base (e.g. 64), not the model's actual input channels (1 or 2).
    #   Using it caused in_ch=64 → size mismatch when loading the checkpoint.
    first_conv_key = None
    out_weight_key = None
    FIRST_CONV_PATTERNS = [
        "inc.net.0.weight",    # UNet_ASPP_Attn (Option B) + standard UNet2D
        "enc1.net.0.weight",   # older UNet2D naming
    ]
    for k in sd.keys():
        if first_conv_key is None:
            for pat in FIRST_CONV_PATTERNS:
                if k == pat or k.endswith("." + pat):
                    first_conv_key = k
                    break
        if k.endswith("out.weight"):
            out_weight_key = k

    if first_conv_key is None or out_weight_key is None:
        # Provide a diagnostic dump to help debug future model types
        raise KeyError(
            f"Could not infer in/out channels from checkpoint keys.\n"
            f"Tried first-conv patterns: {FIRST_CONV_PATTERNS}\n"
            f"State-dict keys (first 15): {list(sd.keys())[:15]}"
        )

    in_ch = int(sd[first_conv_key].shape[1])
    out_ch = int(sd[out_weight_key].shape[0])
    keys = list(sd.keys())

    if any(k.startswith("temporal_attn.") or ".temporal_attn." in k for k in keys):
        from src.obj1.conference.temporal_attn_unet import CTAUNet

        model = CTAUNet(
            in_ch=1,
            out_ch=25,
            base=int(cfg.get("base", 64)),
            t_embed_dim=int(cfg.get("t_embed_dim", 25)),
            t_heads=int(cfg.get("t_heads", 5)),
            pool_stride=int(cfg.get("pool_stride", 4)),
        ).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "multioutput"

    if any(k.startswith("patch_embed.") or ".patch_embed." in k for k in keys):
        from src.obj1.conference.swin_unet import SwinUNet

        model = SwinUNet(
            in_ch=1,
            out_ch=25,
            base_dim=int(cfg.get("base_dim", 96)),
            window_size=int(cfg.get("window_size", 8)),
        ).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "multioutput"

    if any(k.startswith("stages.0.patch_embed.proj.") for k in keys):
        from src.obj1.conference.mixvision_unet import MixVisionUNet

        model = MixVisionUNet(
            in_ch=1,
            out_ch=25,
            base_dim=int(cfg.get("base_dim", 48)),
            decoder_dim=int(cfg.get("decoder_dim", 128)),
        ).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "multioutput"

    if any(k.startswith("wmb.") or ".wmb." in k for k in keys):
        from src.obj1.conference.wmsnet import WMSNet

        # Transport-conditioned WMSNet stores model_family and input_channels in config.json.
        # K-only WMSNet uses in_ch=1; TC variant uses in_ch from config (default 3).
        wmsnet_in_ch = int(cfg.get("input_channels", 1))
        model = WMSNet(
            in_ch=wmsnet_in_ch,
            out_ch=25,
            base=int(cfg.get("base_channels", 64)),
            attn_heads=int(cfg.get("attn_heads", 4)),
        ).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "multioutput"

    if cfg.get("model_family") == "ms_tmo_transport_conditioned":
        from src.obj1.conference.train_unet_multiout_logc_B_aspp_attn import UNet_ASPP_Attn

        model = UNet_ASPP_Attn(
            in_ch=int(cfg.get("input_channels", 3)),
            out_ch=25,
            base=int(cfg.get("base_channels", 64)),
            attn_heads=int(cfg.get("attn_heads", 4)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model, "model", "multioutput"

    # NormSiLU ablation: same 4-stage GN+SiLU backbone naming as Option B,
    # but without ASPP or bottleneck attention modules.
    if (
        any(k.startswith("mid.") for k in keys)
        and any(k.startswith("u3.") for k in keys)
        and not any(k.startswith("aspp.") or ".aspp." in k for k in keys)
        and not any(k.startswith("attn.") or ".attn." in k for k in keys)
    ):
        from src.obj1.conference.train.train_normsilu_baseline import UNet_NormSiLU

        base = int(cfg.get("base_channels", 64))
        model = UNet_NormSiLU(in_ch=in_ch, out_ch=out_ch, base=base).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "multioutput"

    # Optional bottleneck ablations for MS-TMO: ASPP only or attention only.
    has_aspp = any(k.startswith("aspp.") or ".aspp." in k for k in keys)
    has_attn = any(k.startswith("attn.") or ".attn." in k for k in keys)
    if has_aspp and not has_attn:
        from src.obj1.conference.train.train_unet_bottleneck_ablation import UNet_ASPPOnly

        base = int(cfg.get("base_channels", 64))
        model = UNet_ASPPOnly(in_ch=in_ch, out_ch=out_ch, base=base).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "multioutput"

    if has_attn and not has_aspp:
        from src.obj1.conference.train.train_unet_bottleneck_ablation import UNet_AttnOnly

        base = int(cfg.get("base_channels", 64))
        heads = int(cfg.get("attn_heads", 4))
        model = UNet_AttnOnly(in_ch=in_ch, out_ch=out_ch, base=base, attn_heads=heads).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "multioutput"

    # Option B
    if has_attn:
        from src.obj1.conference.train_unet_multiout_logc_B_aspp_attn import UNet_ASPP_Attn
        base = int(cfg.get("base_channels", 64))
        heads = int(cfg.get("attn_heads", 4))
        model = UNet_ASPP_Attn(in_ch=in_ch, out_ch=out_ch, base=base, attn_heads=heads).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, "model", "timecond" if (in_ch == 2 and out_ch == 1) else "multioutput"

    # Plain UNet2D / Option A / baseline / time-conditioned
    from src.obj1.conference.unet2d import UNet2D
    base = int(cfg.get("base_channels", 64))
    model = UNet2D(in_ch=in_ch, out_ch=out_ch, base=base).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    if in_ch == 2 and out_ch == 1:
        return model, "model", "timecond"
    if out_ch == 25:
        return model, "model", "multioutput"

    raise ValueError(f"Unsupported checkpoint shape: in_ch={in_ch}, out_ch={out_ch}")


# =========================================================
# Unified evaluation
# =========================================================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Unified evaluation for Obj1/2/3 patch-based plume surrogates.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--stats_json", required=True)
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--patch", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)

    ap.add_argument("--use_logK", action="store_true")
    ap.add_argument("--eps_k", type=float, default=1e-6)
    ap.add_argument("--eps_c", type=float, default=1e-12)

    ap.add_argument("--plume_thresh", type=float, default=1e-8)
    ap.add_argument("--plume_pad", type=int, default=8)
    ap.add_argument("--plume_min_pixels", type=int, default=64)

    ap.add_argument(
        "--time_mode",
        type=str,
        default="linear_0_1",
        choices=["linear_0_1", "linear_-1_1"],
        help="Time-plane encoding for time-conditioned models.",
    )

    # SSIM protocol
    ap.add_argument(
        "--ssim_data_range",
        type=float,
        default=None,
        help="Fixed SSIM data_range in log space. If omitted, uses ssim_log_max - ssim_log_min.",
    )
    ap.add_argument(
        "--ssim_log_min",
        type=float,
        default=-12.0,
        help="Lower bound of the shared log-concentration evaluation range.",
    )
    ap.add_argument(
        "--ssim_log_max",
        type=float,
        default=2.0,
        help="Upper bound of the shared log-concentration evaluation range.",
    )

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_global_json", default="")
    ap.add_argument("--out_plume_json", default="")

    # Optional prediction-export flags. Default behaviour is unchanged: if
    # --out_pred_dir is empty, no prediction arrays are written.
    ap.add_argument("--out_pred_dir", type=str, default="",
                    help="If set, save per-file full-field predictions as .npz.")
    ap.add_argument("--out_pred_dtype", type=str, default="float16",
                    choices=["float16", "float32"],
                    help="dtype for saved predictions when --out_pred_dir is set.")
    ap.add_argument("--out_pred_compress", action="store_true",
                    help="Use np.savez_compressed instead of np.savez when saving predictions.")
    ap.add_argument("--out_pred_save_target", action="store_true",
                    help="Also save the ground-truth concentration array next to each prediction.")
    ap.add_argument("--out_pred_family", type=str, default="",
                    help="Optional family/model tag recorded in the export manifest.")
    ap.add_argument("--out_pred_seed", type=int, default=-1,
                    help="Optional seed integer recorded in the export manifest.")
    ap.add_argument("--out_pred_source_report", type=str, default="",
                    help="Optional path to the canonical test_report_combined.json for cross-reference.")
    ap.add_argument("--max_test_files", type=int, default=0,
                    help="Optional smoke/debug limit on the number of test files. Default 0 evaluates all files.")
    args = ap.parse_args()

    stats = load_stats(args.stats_json)
    _, _, test_params = load_param_split(args.split_json)
    test_files = flatten_param_folders(args.data_root, test_params)
    if len(test_files) == 0:
        raise RuntimeError("No test files found. Check --data_root and --split_json.")
    if int(args.max_test_files) > 0:
        test_files = test_files[: int(args.max_test_files)]

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = ckpt.get("cfg", {})

    use_logK = bool(cfg.get("use_logK", args.use_logK))
    eps_k = float(cfg.get("eps_k", args.eps_k))
    eps_c = float(cfg.get("eps_c", args.eps_c))
    ssim_data_range = resolve_ssim_data_range(args)

    model, model_key, model_mode = build_model_from_ckpt(ckpt, args.device)
    use_transport_conditioning = bool(cfg.get("use_transport_conditioning", False))
    transport_metadata = None
    condition_params = cfg.get("condition_params", [])
    conditioning_stats = cfg.get("transport_conditioning_stats", {})
    if use_transport_conditioning:
        transport_metadata = load_transport_metadata(str(cfg.get("transport_metadata_csv", "")))

    all_ssim = []
    all_l1 = []
    all_mse = []
    all_mass = []
    all_ssim_plume = []
    all_plume_area_fracs = []

    ssim_per_t = None
    plume_ssim_per_t = None
    plume_area_per_t = None
    cnt_per_t = None
    plume_cnt_per_t = None

    for fp in test_files:
        d = np.load(fp)
        K = d["K"].astype(np.float32)
        C = d["C"].astype(np.float32)  # (T, H, W), physical concentration
        d.close()

        T, H, W = C.shape

        if ssim_per_t is None:
            ssim_per_t = np.zeros(T, dtype=np.float64)
            plume_ssim_per_t = np.zeros(T, dtype=np.float64)
            plume_area_per_t = np.zeros(T, dtype=np.float64)
            cnt_per_t = np.zeros(T, dtype=np.int64)
            plume_cnt_per_t = np.zeros(T, dtype=np.int64)

        K_norm = normalize_K(K, stats, use_logK, eps_k)

        extra_channels = None
        if use_transport_conditioning:
            extra_channels = make_condition_maps(
                param_folder_from_path(fp),
                K_norm.shape[0],
                K_norm.shape[1],
                transport_metadata,
                condition_params,
                conditioning_stats,
            )

        if model_mode == "multioutput":
            pred_log = stitch_predict_allT_multioutput(
                model,
                K_norm,
                args.patch,
                args.stride,
                args.device,
                extra_channels=extra_channels,
            )
        elif model_mode == "timecond":
            pred_log = stitch_predict_allT_timecond(
                model=model,
                K_norm=K_norm,
                patch=args.patch,
                stride=args.stride,
                device=args.device,
                T=T,
                time_mode=args.time_mode,
                extra_channels=extra_channels,
            )
        else:
            raise ValueError(f"Unknown model_mode={model_mode}")

        if pred_log.shape != C.shape:
            raise ValueError(f"Prediction shape {pred_log.shape} does not match GT shape {C.shape} for file {fp}")

        # Optional prediction-export block. No-op unless --out_pred_dir is set.
        if args.out_pred_dir:
            os.makedirs(args.out_pred_dir, exist_ok=True)
            parent = os.path.basename(os.path.dirname(fp))
            base = os.path.splitext(os.path.basename(fp))[0]
            stem = f"{parent}__{base}" if parent else base
            save_path = os.path.join(args.out_pred_dir, stem + ".pred.npz")
            pred_to_save = pred_log.astype(args.out_pred_dtype, copy=False)
            saver = np.savez_compressed if args.out_pred_compress else np.savez
            save_kwargs = {
                "pred_log": pred_to_save,
                "source_file": os.path.basename(fp),
                "T": np.int32(T),
                "H": np.int32(pred_log.shape[1]),
                "W": np.int32(pred_log.shape[2]),
                "patch": np.int32(args.patch),
                "stride": np.int32(args.stride),
                "plume_thresh": np.float64(args.plume_thresh),
                "plume_pad": np.int32(args.plume_pad),
                "plume_min_pixels": np.int32(args.plume_min_pixels),
                "ssim_data_range": np.float64(ssim_data_range),
                "eps_c": np.float64(eps_c),
            }
            if args.out_pred_save_target:
                save_kwargs["gt_phys"] = C.astype(args.out_pred_dtype, copy=False)
            saver(save_path, **save_kwargs)

        for t in range(T):
            gt_phys = np.clip(C[t], 0.0, None)
            gt_log = gt_log10(gt_phys, eps_c)
            pr_log = pred_log[t]

            # Global structural similarity in fixed log-space range
            s_global = ssim_fixed(gt_log, pr_log, data_range=ssim_data_range)
            all_ssim.append(s_global)
            ssim_per_t[t] += s_global

            # Physical-space errors
            pr_phys = log10_to_phys(pr_log, eps_c)
            diff = pr_phys - gt_phys
            all_l1.append(float(np.mean(np.abs(diff))))
            all_mse.append(float(np.mean(diff * diff)))
            all_mass.append(float(np.abs(pr_phys.sum() - gt_phys.sum())))

            # Plume-focused evaluation using GT-derived ROI
            mask = gt_phys > args.plume_thresh
            plume_area = float(mask.mean())
            all_plume_area_fracs.append(plume_area)
            plume_area_per_t[t] += plume_area

            if int(mask.sum()) >= int(args.plume_min_pixels):
                bb = bbox_from_mask(mask, pad=args.plume_pad)
                if bb is not None:
                    y0, y1, x0, x1 = bb
                    if (y1 - y0) >= 7 and (x1 - x0) >= 7:
                        s_plume = ssim_fixed(gt_log[y0:y1, x0:x1], pr_log[y0:y1, x0:x1], data_range=ssim_data_range)
                        all_ssim_plume.append(s_plume)
                        plume_ssim_per_t[t] += s_plume
                        plume_cnt_per_t[t] += 1

            cnt_per_t[t] += 1

    ssim_per_t = (ssim_per_t / np.maximum(cnt_per_t, 1)).tolist()
    plume_ssim_per_t = (plume_ssim_per_t / np.maximum(plume_cnt_per_t, 1)).tolist()
    plume_area_per_t = (plume_area_per_t / np.maximum(cnt_per_t, 1)).tolist()

    global_out = {
        "mean_ssim_log": float(np.mean(all_ssim)),
        "ssim_per_timestep_log": ssim_per_t,
        "sample_count_per_timestep": cnt_per_t.tolist(),
        "mean_l1_phys": float(np.mean(all_l1)),
        "mean_mse_phys": float(np.mean(all_mse)),
        "mean_mass_err_phys": float(np.mean(all_mass)),
        "n_test_files": int(len(test_files)),
        "patch": int(args.patch),
        "stride": int(args.stride),
        "use_logK": bool(use_logK),
        "eps_k": float(eps_k),
        "eps_c": float(eps_c),
        "ssim_data_range": float(ssim_data_range),
    }

    plume_out = {
        "mean_plume_ssim_log_bbox": float(np.mean(all_ssim_plume)) if len(all_ssim_plume) else float("nan"),
        "plume_ssim_per_timestep_log_bbox": plume_ssim_per_t,
        "plume_sample_count_per_timestep": plume_cnt_per_t.tolist(),
        "mean_plume_area_fraction": float(np.mean(all_plume_area_fracs)),
        "plume_area_fraction_per_timestep": plume_area_per_t,
        "n_test_files": int(len(test_files)),
        "patch": int(args.patch),
        "stride": int(args.stride),
        "ssim_data_range": float(ssim_data_range),
        "plume_mask": {
            "mode": "phys_gt_thresh",
            "thresh": float(args.plume_thresh),
            "pad": int(args.plume_pad),
            "min_pixels": int(args.plume_min_pixels),
            "definition": "bbox ROI around GT plume mask",
        },
    }

    combined_out = {
        "protocol": {
            "ckpt": args.ckpt,
            "ckpt_key": model_key,
            "model_mode": model_mode,
            "patch": int(args.patch),
            "stride": int(args.stride),
            "use_logK": bool(use_logK),
            "eps_k": float(eps_k),
            "eps_c": float(eps_c),
            "plume_thresh": float(args.plume_thresh),
            "plume_pad": int(args.plume_pad),
            "plume_min_pixels": int(args.plume_min_pixels),
            "time_mode": args.time_mode,
            "n_test_files": int(len(test_files)),
            "ssim_range_mode": "fixed",
            "ssim_data_range": float(ssim_data_range),
            "ssim_log_min": float(args.ssim_log_min),
            "ssim_log_max": float(args.ssim_log_max),
        },
        "global": global_out,
        "plume": plume_out,
    }

    if args.out_pred_dir:
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            ).decode("ascii").strip()
        except Exception:
            git_hash = ""

        export_manifest = {
            "schema": "obj1_pred_export_v1",
            "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "family": args.out_pred_family,
            "seed": int(args.out_pred_seed),
            "ckpt": args.ckpt,
            "split_json": args.split_json,
            "stats_json": args.stats_json,
            "data_root": args.data_root,
            "model_mode": model_mode,
            "ckpt_key": model_key,
            "patch": int(args.patch),
            "stride": int(args.stride),
            "use_logK": bool(use_logK),
            "eps_k": float(eps_k),
            "eps_c": float(eps_c),
            "plume_thresh": float(args.plume_thresh),
            "plume_pad": int(args.plume_pad),
            "plume_min_pixels": int(args.plume_min_pixels),
            "ssim_data_range": float(ssim_data_range),
            "pred_shape_TxHxW": [int(T), int(H), int(W)] if (cnt_per_t is not None) else None,
            "out_pred_dtype": args.out_pred_dtype,
            "out_pred_compress": bool(args.out_pred_compress),
            "out_pred_save_target": bool(args.out_pred_save_target),
            "n_test_files": int(len(test_files)),
            "test_files": [os.path.basename(p) for p in test_files],
            "source_report": args.out_pred_source_report,
            "git_hash": git_hash,
            "out_pred_dir": os.path.abspath(args.out_pred_dir),
        }
        manifest_path = os.path.join(args.out_pred_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(export_manifest, f, indent=2)
        print("[Saved export manifest]", manifest_path)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(combined_out, f, indent=2)
    print("[Saved combined]", args.out_json)

    if args.out_global_json.strip():
        os.makedirs(os.path.dirname(args.out_global_json) or ".", exist_ok=True)
        with open(args.out_global_json, "w", encoding="utf-8") as f:
            json.dump(global_out, f, indent=2)
        print("[Saved global]", args.out_global_json)

    if args.out_plume_json.strip():
        os.makedirs(os.path.dirname(args.out_plume_json) or ".", exist_ok=True)
        with open(args.out_plume_json, "w", encoding="utf-8") as f:
            json.dump(plume_out, f, indent=2)
        print("[Saved plume]", args.out_plume_json)

    print(json.dumps(combined_out, indent=2))


if __name__ == "__main__":
    main()

# For TMO-Unet/ Opt A/B / Time-cond UNet

# python -m src.eval.eval_obj1_unified_logc_global_plumemask ^
#   --data_root "./data/T25_TSTEP_OVERRIDE_FINAL" ^
#   --split_json "splits/param_split_fixed.json" ^
#   --stats_json "stats/train_stats.json" ^
#   --ckpt "runs/obj1_fno_multiout_logc_p320_w64_m20x20_d4_lr0.0001/seed0/best.pt" ^
#   --patch 320 --stride 160 --use_logK ^
#   --out_json "runs/obj1_fno_multiout_logc_p320_w64_m20x20_d4_lr0.0001/seed0/test_report_combined.json" ^
#   --out_global_json "runs/obj1_fno_multiout_logc_p320_w64_m20x20_d4_lr0.0001/seed0/test_report_global.json" ^
#   --out_plume_json "runs/obj1_fno_multiout_logc_p320_w64_m20x20_d4_lr0.0001/seed0/test_report_plumemask.json"



# Pix2pix

# python -m src.eval.eval_obj1_unified_logc_global_plumemask ^
#   --data_root "./data/T25_TSTEP_OVERRIDE_FINAL" ^
#   --split_json "splits/param_split_fixed.json" ^
#   --stats_json "stats/train_stats.json" ^
#   --ckpt "runs/obj1_pix2pix_multiout_logc_p320_lr0.0001/seed0/best.pt" ^
#   --patch 320 --stride 160 --use_logK ^
#   --out_json "runs/obj1_pix2pix_multiout_logc_p320_lr0.0001/seed0/test_report_combined.json" ^
#   --out_global_json "runs/obj1_pix2pix_multiout_logc_p320_lr0.0001/seed0/test_report_global.json" ^
#   --out_plume_json "runs/obj1_pix2pix_multiout_logc_p320_lr0.0001/seed0/test_report_plumemask.json"
