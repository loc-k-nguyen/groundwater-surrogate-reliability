"""Certify Obj3 prediction caches against the sliding-boundary failure signature."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REQUIRED_KEYS = ("y_true", "y_pred", "y_pred_std", "c_phys")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def old_coverage_end(length: int, patch: int, stride: int) -> int:
    starts = list(range(0, length - patch + 1, stride))
    if not starts:
        return length
    return starts[-1] + patch


def validate_array(
    array: np.ndarray,
    *,
    label: str,
    expected_shape: tuple[int, int, int, int],
) -> dict[str, float | int | list[int]]:
    if array.shape != expected_shape:
        raise ValueError(f"{label}: expected shape {expected_shape}, found {array.shape}")

    minimum = float("inf")
    maximum = float("-inf")
    exact_zeros = 0
    for sample_index in range(array.shape[0]):
        sample = np.asarray(array[sample_index])
        if not np.isfinite(sample).all():
            raise ValueError(f"{label}: nonfinite value in sample {sample_index}")
        minimum = min(minimum, float(sample.min()))
        maximum = max(maximum, float(sample.max()))
        exact_zeros += int(np.count_nonzero(sample == 0.0))

    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": minimum,
        "max": maximum,
        "exact_zero_count": exact_zeros,
    }


def validate_boundary_signature(
    array: np.ndarray,
    *,
    label: str,
    patch: int,
    stride: int,
) -> dict[str, int]:
    row_end = old_coverage_end(array.shape[2], patch, stride)
    col_end = old_coverage_end(array.shape[3], patch, stride)
    lower_uniform_zero = 0
    right_uniform_zero = 0

    for sample_index in range(array.shape[0]):
        sample = np.asarray(array[sample_index])
        if row_end < array.shape[2]:
            lower_uniform_zero += int(
                np.count_nonzero(np.all(sample[:, row_end:, :] == 0.0, axis=(1, 2)))
            )
        if col_end < array.shape[3]:
            right_uniform_zero += int(
                np.count_nonzero(np.all(sample[:, :, col_end:] == 0.0, axis=(1, 2)))
            )

    if lower_uniform_zero or right_uniform_zero:
        raise ValueError(
            f"{label}: stale boundary signature found; "
            f"lower_uniform_zero={lower_uniform_zero}, right_uniform_zero={right_uniform_zero}"
        )
    return {
        "old_row_coverage_end": row_end,
        "old_col_coverage_end": col_end,
        "lower_uniform_zero_sample_timesteps": lower_uniform_zero,
        "right_uniform_zero_sample_timesteps": right_uniform_zero,
    }


def certify_cache(
    label: str,
    path: Path,
    expected_samples: int,
    patch: int,
    stride: int,
) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label}: missing or empty cache: {path}")

    record: dict[str, object] = {
        "label": label,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrays": {},
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(set(REQUIRED_KEYS) - set(payload.files))
        if missing:
            raise KeyError(f"{label}: missing arrays {missing}")

        prediction_shape: tuple[int, int, int, int] | None = None
        for key in REQUIRED_KEYS:
            array = payload[key]
            if array.ndim != 4:
                raise ValueError(f"{label}/{key}: expected four dimensions, found {array.shape}")
            if prediction_shape is None:
                prediction_shape = tuple(int(value) for value in array.shape)
                if prediction_shape[0] != expected_samples:
                    raise ValueError(
                        f"{label}: expected {expected_samples} samples, found {prediction_shape[0]}"
                    )
                if prediction_shape[1:] != (25, 600, 400):
                    raise ValueError(
                        f"{label}: expected channels/field (25, 600, 400), found {prediction_shape[1:]}"
                    )
            stats = validate_array(
                array,
                label=f"{label}/{key}",
                expected_shape=prediction_shape,
            )
            if key in {"y_pred", "y_pred_std"}:
                stats["boundary_signature"] = validate_boundary_signature(
                    array,
                    label=f"{label}/{key}",
                    patch=patch,
                    stride=stride,
                )
            record["arrays"][key] = stats
            del array
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        nargs=3,
        action="append",
        metavar=("LABEL", "PATH", "EXPECTED_SAMPLES"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch", type=int, default=320)
    parser.add_argument("--stride", type=int, default=160)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite certification report: {args.output}")

    records = [
        certify_cache(label, Path(path), int(expected), args.patch, args.stride)
        for label, path, expected in args.cache
    ]
    report = {
        "status": "BOUNDARY_FIXED_CACHE_CERTIFIED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch": args.patch,
        "stride": args.stride,
        "caches": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"BOUNDARY_FIXED_CACHE_CERTIFIED {args.output}")


if __name__ == "__main__":
    main()
