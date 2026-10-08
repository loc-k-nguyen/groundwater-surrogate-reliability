import torch
import torch.nn as nn
import torch.nn.functional as F

from src.obj1.conference.fno2d import _meshgrid_ij


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONet2D(nn.Module):
    """
    Fixed-grid DeepONet variant for multi-output plume surrogates.
    Branch net encodes K(x,y), trunk net encodes spatial coordinates, and the
    dot-product basis is reshaped to (T, H, W).
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
            nn.Linear(branch_latent, out_ch * basis_rank),
        )

        self.trunk = nn.Sequential(
            nn.Linear(2, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, basis_rank),
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def _coord_grid(self, x: torch.Tensor) -> torch.Tensor:
        _, _, hgt, wid = x.shape
        yy = torch.linspace(0.0, 1.0, steps=hgt, device=x.device, dtype=x.dtype)
        xx = torch.linspace(0.0, 1.0, steps=wid, device=x.device, dtype=x.dtype)
        gy, gx = _meshgrid_ij(yy, xx)
        coords = torch.stack([gx, gy], dim=-1).view(-1, 2)
        return coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, hgt, wid = x.shape
        branch = self.branch(x.float()).view(bsz, self.out_ch, self.basis_rank)
        trunk = self.trunk(self._coord_grid(x)).view(hgt * wid, self.basis_rank)
        out = torch.einsum("btr,nr->btn", branch, trunk) + self.bias.view(1, self.out_ch, 1)
        return out.view(bsz, self.out_ch, hgt, wid)
