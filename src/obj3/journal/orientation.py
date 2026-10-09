"""Explicit evaluation-only spatial alignment without rewriting stored datasets."""
from pathlib import Path
from typing import Sequence

import torch

from src.obj3.conference.data_obj3_ood import Obj3FullFieldDataset


def reverse_sample_rows(sample):
    """Reverse input and both target representations together, leaving metadata intact."""
    aligned = dict(sample)
    for key in ("K", "C_log", "C_phys"):
        value = sample[key]
        if value.ndim != 3 or tuple(value.shape[-2:]) != (600, 400):
            raise ValueError(f"Unexpected {key} geometry: {tuple(value.shape)}")
        aligned[key] = torch.flip(value, dims=(-2,))
    return aligned


class OrientationAlignedFullFieldDataset(Obj3FullFieldDataset):
    """Select a declared stored orientation per root; never infer it from predictions."""

    def __init__(self, file_list, stats, *, native_roots: Sequence[Path],
                 reverse_roots: Sequence[Path], **kwargs):
        super().__init__(file_list, stats, **kwargs)
        native = [Path(p).resolve() for p in native_roots]
        reverse = [Path(p).resolve() for p in reverse_roots]
        if not native or not reverse:
            raise ValueError("Both native and reverse root sets must be declared")
        self.reverse_flags = []
        for path in self.files:
            resolved = Path(path).resolve()
            matches = ([False for root in native if resolved.is_relative_to(root)] +
                       [True for root in reverse if resolved.is_relative_to(root)])
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one orientation root for {path}; got {matches}")
            self.reverse_flags.append(matches[0])

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return reverse_sample_rows(sample) if self.reverse_flags[index] else sample
