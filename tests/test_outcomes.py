import pandas as pd

from oulad.outcomes import mean_batch_auc, summarize, summarize_by_group


def _df():
    return pd.DataFrame({
        "batch": ["a"] * 6,
        "at_risk": [1, 1, 1, 1, 0, 0],
        "flagged": [1, 1, 1, 0, 1, 0],
        "risk_score": [0.9, 0.8, 0.7, 0.2, 0.6, 0.1],
        "disability": ["N", "N", "Y", "Y", "N", "Y"],
    })


def test_summarize_counts():
    s = summarize(_df())
    assert s["students"] == 6 and s["at_risk"] == 4
    assert s["flagged"] == 4 and s["caught"] == 3
    assert s["recall"] == 0.75 and s["precision"] == 0.75


def test_contacts_saved_vs_random_list():
    s = summarize(_df())
    # flags 4/6 = 0.667 of students to reach 0.75 of at-risk; random would need 0.75
    assert abs(s["contacts_saved"] - (1 - (4 / 6) / 0.75)) < 1e-9


def test_by_group():
    g = summarize_by_group(_df())
    assert g["N"]["students"] == 3 and g["Y"]["students"] == 3
    assert g["Y"]["recall"] == 1 / 2


def test_mean_batch_auc_skips_single_class_batches():
    df = pd.concat([_df(), _df().assign(batch="b", at_risk=1)])
    assert mean_batch_auc(df) == mean_batch_auc(_df())
