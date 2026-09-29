"""Population Stability Index drift checks, a drift injector, and an Evidently HTML report."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
EPS = 1e-4


def _psi(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _as_float(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref_vals = _as_float(ref)
    ref_clean = ref_vals[~np.isnan(ref_vals)]
    if len(ref_clean) == 0:
        return float("nan")
    # Inner decile edges. np.unique collapses repeated edges (e.g. many zeros).
    inner = np.unique(np.quantile(ref_clean, np.linspace(0, 1, bins + 1)[1:-1]))

    def dist(values: np.ndarray) -> np.ndarray:
        na = np.isnan(values)
        idx = np.searchsorted(inner, values[~na], side="right")
        counts = np.bincount(idx, minlength=len(inner) + 1).astype(float)
        counts = np.append(counts, na.sum())          # last bin = missing
        return counts / max(counts.sum(), 1.0)

    return _psi(dist(ref_vals), dist(_as_float(cur)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.astype("string").fillna("__missing__")
    c = cur.astype("string").fillna("__missing__")
    cats = sorted(set(r) | set(c))
    p = r.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    q = c.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(p, q)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    psi_threshold: float,
) -> dict:
    psi = {c: psi_numeric(reference[c], current[c]) for c in numeric}
    psi.update({c: psi_categorical(reference[c], current[c]) for c in categorical})
    drifted = sorted(c for c, v in psi.items() if not np.isnan(v) and v > psi_threshold)
    return {
        "psi": {c: round(v, 4) for c, v in psi.items()},
        "drifted": drifted,
        "drift_share": len(drifted) / max(len(psi), 1),
        "n_features": len(psi),
    }


def simulate_engagement_drop(
    df: pd.DataFrame, share: float = 0.4, factor: float = 0.3, seed: int = 0
) -> pd.DataFrame:
    """Inject a known shift: a share of students lose most of their recent activity,
    as if the VLE had an outage. Used to prove the monitor fires."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    hit = rng.random(len(out)) < share
    for col in ["clicks_total", "clicks_last_7d", "active_days", "clicks_oucontent", "clicks_quiz"]:
        out.loc[hit, col] = (out.loc[hit, col] * factor).round()
    out["click_trend"] = np.log1p(out["clicks_last_7d"]) - np.log1p(out["clicks_prev_7d"])
    return out


def evidently_report_html(reference: pd.DataFrame, current: pd.DataFrame) -> bytes | None:
    """Evidently data drift report as HTML bytes, or None if Evidently is unavailable."""
    path = Path(tempfile.mkdtemp()) / "drift.html"
    try:
        try:  # Evidently >= 0.7
            from evidently import Report
            from evidently.presets import DataDriftPreset

            Report([DataDriftPreset()]).run(
                current_data=current, reference_data=reference
            ).save_html(str(path))
        except ImportError:  # Evidently 0.4.x - 0.6.x
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)
            report.save_html(str(path))
        return path.read_bytes()
    except Exception as exc:  # noqa: BLE001  the report is a nice-to-have, never a blocker
        log.warning("Evidently report skipped: %s", exc)
        return None
