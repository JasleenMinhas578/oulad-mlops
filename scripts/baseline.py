"""Logistic regression baseline on the same grouped split the LightGBM model uses."""
import time
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oulad.data import TRAIN_TERMS
from oulad.features import LABEL, NUMERIC_FEATURES, build_features
from oulad.train import train_model

df = build_features(Path("data/bronze"), 28)
df = df[df["code_presentation"].isin(TRAIN_TERMS)].reset_index(drop=True)
tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
    df, groups=df["id_student"]))
tr, va = df.iloc[tr_idx], df.iloc[va_idx]

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
     NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["code_module"]),
])
logreg = make_pipeline(pre, LogisticRegression(max_iter=2000)).fit(tr, tr[LABEL])
print(f"logistic regression AUC = {roc_auc_score(va[LABEL], logreg.predict_proba(va)[:, 1]):.3f}")

t = time.perf_counter()
trained, v = train_model(df, False, 0.8)
print(f"lightgbm AUC = {roc_auc_score(v[LABEL], v['proba']):.3f}, training time {time.perf_counter() - t:.1f}s")
