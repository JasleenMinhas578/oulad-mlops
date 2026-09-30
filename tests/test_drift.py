import numpy as np
import pandas as pd

from oulad.drift import detect_drift, psi_categorical, psi_numeric, simulate_engagement_drop


def test_psi_identical_is_near_zero():
    x = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi_numeric(x, x.copy()) < 0.01


def test_psi_detects_mean_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(1, 1, 5000))
    assert psi_numeric(ref, cur) > 0.2


def test_psi_counts_missingness_as_drift():
    ref = pd.Series([1.0] * 900 + [np.nan] * 100)
    cur = pd.Series([1.0] * 500 + [np.nan] * 500)
    assert psi_numeric(ref, cur) > 0.2


def test_categorical_psi_flags_unseen_category():
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    cur = pd.Series(["C"] * 1000)
    assert psi_categorical(ref, cur) > 1.0


def test_detect_drift_reports_share():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(
        {"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000), "m": ["X"] * 2000}
    )
    cur = pd.DataFrame(
        {"a": rng.normal(2, 1, 2000), "b": rng.normal(0, 1, 2000), "m": ["X"] * 2000}
    )
    rep = detect_drift(ref, cur, ["a", "b"], ["m"], psi_threshold=0.2)
    assert rep["drifted"] == ["a"]
    assert abs(rep["drift_share"] - 1 / 3) < 1e-9


def test_injected_engagement_drop_is_caught():
    rng = np.random.default_rng(2)
    n = 3000
    df = pd.DataFrame({
        "clicks_total": rng.poisson(200, n).astype(float),
        "clicks_last_7d": rng.poisson(50, n).astype(float),
        "clicks_prev_7d": rng.poisson(50, n).astype(float),
        "active_days": rng.poisson(15, n).astype(float),
        "clicks_oucontent": rng.poisson(80, n).astype(float),
        "clicks_quiz": rng.poisson(30, n).astype(float),
    })
    df["click_trend"] = np.log1p(df["clicks_last_7d"]) - np.log1p(df["clicks_prev_7d"])
    rep = detect_drift(
        df, simulate_engagement_drop(df), ["clicks_total", "clicks_last_7d"], [], 0.2
    )
    assert "clicks_total" in rep["drifted"]


def test_reference_uses_same_course_when_available():
    from oulad.drift import reference_for_course

    ref = pd.DataFrame({"code_module": ["A"] * 300 + ["B"] * 300, "x": range(600)})
    cur = pd.DataFrame({"code_module": ["A"] * 10, "x": range(10)})
    used, scope = reference_for_course(ref, cur)
    assert scope == "same course" and set(used["code_module"]) == {"A"}


def test_reference_falls_back_for_new_course():
    from oulad.drift import reference_for_course

    ref = pd.DataFrame({"code_module": ["A"] * 300, "x": range(300)})
    cur = pd.DataFrame({"code_module": ["Z"] * 10, "x": range(10)})
    used, scope = reference_for_course(ref, cur)
    assert scope == "no reference for this course" and len(used) == 300


def test_activity_outage_only_reports_volume_features():
    from oulad.drift import activity_outage

    psi = {"clicks_total": 0.5, "active_days": 0.1, "clicks_quiz": 0.9}
    assert activity_outage(psi, 0.2) == ["clicks_total"]
