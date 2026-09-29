"""Behavioural features computed strictly from events before the cutoff day."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from oulad.data import KEYS, PRESENTATION, build_population, load_tables

ACTIVITY_TYPES = ["oucontent", "forumng", "homepage", "quiz", "resource", "subpage", "url"]

NUMERIC_FEATURES = [
    "clicks_total", "active_days", "clicks_pre_start",
    "clicks_last_7d", "clicks_prev_7d", "click_trend",
    "days_since_last_active", "never_active",
    *[f"clicks_{a}" for a in ACTIVITY_TYPES], "clicks_other",
    "n_due", "n_submitted", "n_missed", "n_late",
    "mean_score", "score_trend", "mean_days_early",
    "date_registration", "num_of_prev_attempts", "studied_credits",
]
CATEGORICAL_FEATURES = ["code_module"]
DEMOGRAPHIC_FEATURES = ["gender", "age_band", "region", "highest_education", "imd_band"]
AUDIT_COLUMNS = ["disability", "gender", "age_band", "region", "imd_band"]
LABEL = "at_risk"


def feature_columns(include_demographics: bool) -> tuple[list[str], list[str]]:
    categorical = CATEGORICAL_FEATURES + (DEMOGRAPHIC_FEATURES if include_demographics else [])
    return list(NUMERIC_FEATURES), categorical


def vle_features(
    bronze_dir: Path, cutoff_day: int, presentations: list[tuple[str, str]] | None = None
) -> pd.DataFrame:
    sv_path = (Path(bronze_dir) / "student_vle.parquet").as_posix()
    vle_path = (Path(bronze_dir) / "vle.parquet").as_posix()
    con = duckdb.connect()
    con.execute("SET memory_limit = '1GB'")

    join_wanted = ""
    if presentations is not None:
        con.register("wanted", pd.DataFrame(presentations, columns=PRESENTATION))
        join_wanted = (
            "JOIN wanted w ON sv.code_module = w.code_module "
            "AND sv.code_presentation = w.code_presentation"
        )

    activity_sums = ",\n".join(
        f"SUM(CASE WHEN activity_type = '{a}' THEN sum_click ELSE 0 END)::DOUBLE AS clicks_{a}"
        for a in ACTIVITY_TYPES
    )
    known = ", ".join(f"'{a}'" for a in ACTIVITY_TYPES)

    sql = f"""
    WITH v AS (
        SELECT sv.code_module, sv.code_presentation, sv.id_student, sv.date, sv.sum_click,
               COALESCE(r.activity_type, 'other') AS activity_type
        FROM read_parquet('{sv_path}') AS sv
        {join_wanted}
        LEFT JOIN read_parquet('{vle_path}') AS r ON sv.id_site = r.id_site
        WHERE sv.date < $cutoff
    )
    SELECT code_module, code_presentation, id_student,
           SUM(sum_click)::DOUBLE AS clicks_total,
           COUNT(DISTINCT date) AS active_days,
           SUM(CASE WHEN date < 0 THEN sum_click ELSE 0 END)::DOUBLE AS clicks_pre_start,
           SUM(CASE WHEN date >= $cutoff - 7 THEN sum_click ELSE 0 END)::DOUBLE AS clicks_last_7d,
           SUM(CASE WHEN date >= $cutoff - 14 AND date < $cutoff - 7
                    THEN sum_click ELSE 0 END)::DOUBLE AS clicks_prev_7d,
           MAX(date) AS last_active_day,
           {activity_sums},
           SUM(CASE WHEN activity_type NOT IN ({known})
                    THEN sum_click ELSE 0 END)::DOUBLE AS clicks_other
    FROM v
    GROUP BY code_module, code_presentation, id_student
    """
    df = con.execute(sql, {"cutoff": cutoff_day}).df()
    df["click_trend"] = np.log1p(df["clicks_last_7d"]) - np.log1p(df["clicks_prev_7d"])
    df["days_since_last_active"] = cutoff_day - df["last_active_day"]
    return df.drop(columns=["last_active_day"])


def assessment_features(
    tables: dict[str, pd.DataFrame], cutoff_day: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = tables["assessments"]
    a = a[(a["assessment_type"] != "Exam") & a["date"].notna()]
    n_due = (
        a[a["date"] < cutoff_day].groupby(PRESENTATION).size().rename("n_due").reset_index()
    )

    sa = tables["student_assessment"]
    sa = sa[(sa["is_banked"] == 0) & (sa["date_submitted"] < cutoff_day)]
    sa = sa.merge(a[["id_assessment", *PRESENTATION, "date"]], on="id_assessment", how="inner")
    sa["days_early"] = sa["date"] - sa["date_submitted"]
    sa["late"] = (sa["days_early"] < 0).astype(int)
    sa["due_by_cutoff"] = (sa["date"] < cutoff_day).astype(int)
    sa = sa.sort_values("date_submitted")

    out = (
        sa.groupby(KEYS)
        .agg(
            n_submitted=("id_assessment", "nunique"),
            n_submitted_due=("due_by_cutoff", "sum"),
            mean_score=("score", "mean"),
            first_score=("score", "first"),
            last_score=("score", "last"),
            mean_days_early=("days_early", "mean"),
            n_late=("late", "sum"),
        )
        .reset_index()
    )
    out["score_trend"] = np.where(
        out["n_submitted"] >= 2, out["last_score"] - out["first_score"], np.nan
    )
    return out.drop(columns=["first_score", "last_score"]), n_due


def build_features(
    bronze_dir: Path, cutoff_day: int, presentations: list[tuple[str, str]] | None = None
) -> pd.DataFrame:
    """One row per enrolled student: keys, features, audit columns, label."""
    tables = load_tables(bronze_dir)
    pop = build_population(tables, cutoff_day, presentations)
    vle = vle_features(bronze_dir, cutoff_day, presentations)
    assess, n_due = assessment_features(tables, cutoff_day)

    df = (
        pop.merge(vle, on=KEYS, how="left")
        .merge(assess, on=KEYS, how="left")
        .merge(n_due, on=PRESENTATION, how="left")
    )
    zero_if_missing = (
        [c for c in vle.columns if c.startswith("clicks_")]
        + ["active_days", "click_trend", "n_due", "n_submitted", "n_submitted_due", "n_late"]
    )
    df[zero_if_missing] = df[zero_if_missing].fillna(0)
    df["never_active"] = df["days_since_last_active"].isna().astype(int)
    df["n_missed"] = (df["n_due"] - df["n_submitted_due"]).clip(lower=0)

    columns = KEYS + NUMERIC_FEATURES + ["highest_education"] + AUDIT_COLUMNS + ["final_result", LABEL]
    return df[columns]
