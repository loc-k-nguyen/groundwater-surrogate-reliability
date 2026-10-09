"""Descriptive physical-quality diagnostics; integrated concentration is not mass balance."""
import numpy as np


def quality_by_time(y_true, y_pred, c_phys, threshold=1e-8):
    if y_true.shape != y_pred.shape or y_true.shape != c_phys.shape or y_true.ndim != 3:
        raise ValueError("Expected aligned (time, row, column) arrays")
    if not all(np.isfinite(a).all() for a in (y_true, y_pred, c_phys)):
        raise ValueError("Nonfinite metric inputs")
    rows = []
    rr, cc = np.indices(y_true.shape[-2:], dtype=np.float64)
    for t in range(len(y_true)):
        true = np.maximum(c_phys[t].astype(np.float64), 0)
        pred = np.maximum(10.0 ** y_pred[t].astype(np.float64) - 1e-12, 0)
        tm, pm = true > threshold, pred > threshold
        diff = y_pred[t].astype(np.float64) - y_true[t].astype(np.float64)
        ts, ps = true.sum(), pred.sum()
        signed_sum = float((ps - ts) / max(ts, 1e-12))
        tw, pw = np.where(tm, true, 0), np.where(pm, pred, 0)
        centroid = None
        if tw.sum() > 0 and pw.sum() > 0:
            tr, tc = (rr * tw).sum() / tw.sum(), (cc * tw).sum() / tw.sum()
            pr, pc = (rr * pw).sum() / pw.sum(), (cc * pw).sum() / pw.sum()
            centroid = float(np.hypot(pr - tr, pc - tc))
        rows.append({"timestep": t, "log_mae": float(np.abs(diff).mean()),
                     "log_rmse": float(np.sqrt(np.mean(diff ** 2))), "signed_log_bias": float(diff.mean()),
                     "signed_relative_integrated_concentration_error": signed_sum,
                     "absolute_relative_integrated_concentration_error": abs(signed_sum),
                     "signed_relative_peak_error": float((pred.max() - true.max()) / max(true.max(), 1e-12)),
                     "signed_plume_area_error_pixels": int(pm.sum()) - int(tm.sum()),
                     "absolute_plume_area_error_pixels": abs(int(pm.sum()) - int(tm.sum())),
                     "centroid_error_pixels": centroid, "true_positive_pixels": int((tm & pm).sum()),
                     "false_negative_pixels": int((tm & ~pm).sum()), "false_positive_pixels": int((~tm & pm).sum()),
                     "true_plume_pixels": int(tm.sum()), "predicted_plume_pixels": int(pm.sum()),
                     "negative_reference_pixels": int((c_phys[t] < 0).sum())})
    return rows
