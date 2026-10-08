from __future__ import annotations

import torch
import torch.nn.functional as F

from src.obj1.conference.fno2d import FNO2D


class FNOReplicate2D(FNO2D):
    """
    FNO baseline with replicate padding instead of zero padding on the spatial axes.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_coords:
            x = torch.cat([x, self._coord_grid(x)], dim=1)

        x = self.input_proj(x.float())

        pad_h = int(round(x.shape[-2] * self.pad_ratio))
        pad_w = int(round(x.shape[-1] * self.pad_ratio))
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        for block in self.blocks:
            x = block(x)

        if pad_h > 0 or pad_w > 0:
            x = x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]

        return self.head(x)
