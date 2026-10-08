import torch
import torch.nn as nn
import torch.nn.functional as F


def _meshgrid_ij(x: torch.Tensor, y: torch.Tensor):
    try:
        return torch.meshgrid(x, y, indexing="ij")
    except TypeError:
        return torch.meshgrid(x, y)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")

        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        modes1 = min(self.modes1, height)
        modes2 = min(self.modes2, width // 2 + 1)
        if modes1 > 0 and modes2 > 0:
            weight = self.weight[:, :, :modes1, :modes2]
            out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
                x_ft[:, :, :modes1, :modes2], weight
            )

        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x.float())
        y = self.norm(y)
        return F.gelu(y)


class FNO2D(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 25,
        width: int = 64,
        modes1: int = 20,
        modes2: int = 20,
        depth: int = 4,
        use_coords: bool = True,
        pad_ratio: float = 0.125,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.width = width
        self.use_coords = use_coords
        self.pad_ratio = pad_ratio

        lift_in_ch = in_ch + (2 if use_coords else 0)
        self.input_proj = nn.Conv2d(lift_in_ch, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes1, modes2) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_ch, kernel_size=1),
        )

    def _coord_grid(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        yy = torch.linspace(0.0, 1.0, steps=height, device=x.device, dtype=x.dtype)
        xx = torch.linspace(0.0, 1.0, steps=width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = _meshgrid_ij(yy, xx)
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        return grid.expand(bsz, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_coords:
            x = torch.cat([x, self._coord_grid(x)], dim=1)

        x = self.input_proj(x.float())

        pad_h = int(round(x.shape[-2] * self.pad_ratio))
        pad_w = int(round(x.shape[-1] * self.pad_ratio))
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        for block in self.blocks:
            x = block(x)

        if pad_h > 0 or pad_w > 0:
            x = x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]

        return self.head(x)
