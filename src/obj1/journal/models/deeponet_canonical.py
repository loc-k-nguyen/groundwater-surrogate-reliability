from __future__ import annotations

import torch
import torch.nn as nn

from src.obj1.conference.deeponet2d import ConvBlock


class CanonicalDeepONet2D(nn.Module):
    """
    Canonical fixed-grid DeepONet for Obj1.

    The branch encodes the conductivity field K(x, y).
    The trunk encodes full (x, y, t) queries so the spatial basis can vary with time.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 25,
        branch_width: int = 128,
        branch_latent: int = 384,
        trunk_width: int = 384,
        basis_rank: int = 96,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.branch_width = branch_width
        self.branch_latent = branch_latent
        self.trunk_width = trunk_width
        self.basis_rank = basis_rank

        self.branch = nn.Sequential(
            ConvBlock(in_ch, branch_width),
            nn.AvgPool2d(2),
            ConvBlock(branch_width, 2 * branch_width),
            nn.AvgPool2d(2),
            ConvBlock(2 * branch_width, 4 * branch_width),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * branch_width, branch_latent),
            nn.GELU(),
            nn.Linear(branch_latent, basis_rank),
        )

        self.trunk = nn.Sequential(
            nn.Linear(3, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, basis_rank),
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def _spatial_grid(self, x: torch.Tensor) -> torch.Tensor:
        _, _, hgt, wid = x.shape
        yy = torch.linspace(0.0, 1.0, steps=hgt, device=x.device, dtype=x.dtype)
        xx = torch.linspace(0.0, 1.0, steps=wid, device=x.device, dtype=x.dtype)
        try:
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        except TypeError:
            gy, gx = torch.meshgrid(yy, xx)
        return torch.stack([gx, gy], dim=-1).view(hgt * wid, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, hgt, wid = x.shape
        branch = self.branch(x.float()).view(bsz, self.basis_rank)
        coords = self._spatial_grid(x)
        t_values = torch.linspace(0.0, 1.0, steps=self.out_ch, device=x.device, dtype=x.dtype)
        outputs = []
        for t_idx, t_val in enumerate(t_values):
            time_column = torch.full((coords.shape[0], 1), t_val, device=x.device, dtype=x.dtype)
            trunk_in = torch.cat([coords, time_column], dim=-1)
            trunk = self.trunk(trunk_in).view(hgt * wid, self.basis_rank)
            out_t = torch.einsum("br,nr->bn", branch, trunk) + self.bias[t_idx].view(1, 1)
            outputs.append(out_t.view(bsz, 1, hgt, wid))
        return torch.cat(outputs, dim=1)
