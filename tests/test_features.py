import numpy as np
import pandas as pd

from oulad.features import assessment_features, vle_features

P = {"code_module": "AAA", "code_presentation": "2013J"}


def test_assessment_features_respect_cutoff():
    tables = {
        "assessments": pd.DataFrame({
            "id_assessment": [1, 2, 3], **{k: [v] * 3 for k, v in P.items()},
            "assessment_type": ["TMA", "TMA", "Exam"],
            "date": [10, 40, np.nan], "weight": [20, 20, 60],
        }),
        "student_assessment": pd.DataFrame({
            "id_assessment": [1, 2, 2],
            "id_student": [7, 7, 8],
            "date_submitted": [12, 20, 30],   # student 8 submits after the cutoff
            "is_banked": [0, 0, 0],
            "score": [60.0, 80.0, 90.0],
        }),
    }
    out, n_due = assessment_features(tables, cutoff_day=28)
    row = out.set_index("id_student").loc[7]
    assert row["n_submitted"] == 2
    assert row["n_submitted_due"] == 1           # only assessment 1 was due before day 28
    assert row["n_late"] == 1                    # submitted day 12 for a day-10 deadline
    assert row["score_trend"] == 20
    assert row["mean_days_early"] == 9           # (-2 + 20) / 2
    assert 8 not in set(out["id_student"])       # future submission ignored: no leakage
    assert n_due["n_due"].tolist() == [1]


def test_vle_features_ignore_events_after_cutoff(tmp_path):
    pd.DataFrame({
        **{k: [v] * 3 for k, v in P.items()},
        "id_student": [7, 7, 7], "id_site": [1, 2, 1],
        "date": [-5, 25, 30], "sum_click": [3, 4, 100],
    }).to_parquet(tmp_path / "student_vle.parquet")
    pd.DataFrame({
        "id_site": [1, 2], **{k: [v] * 2 for k, v in P.items()},
        "activity_type": ["forumng", "quiz"], "week_from": [None, None], "week_to": [None, None],
    }).to_parquet(tmp_path / "vle.parquet")

    row = vle_features(tmp_path, cutoff_day=28).iloc[0]
    assert row["clicks_total"] == 7              # the 100 clicks on day 30 are the future
    assert row["clicks_pre_start"] == 3
    assert row["clicks_last_7d"] == 4
    assert row["active_days"] == 2
    assert row["clicks_forumng"] == 3 and row["clicks_quiz"] == 4
    assert row["days_since_last_active"] == 3
