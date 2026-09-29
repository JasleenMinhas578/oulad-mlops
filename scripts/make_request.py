"""Print a /predict payload built from three reference rows."""
import json
import sys
from pathlib import Path

import pandas as pd

from oulad import storage

meta = json.loads(Path("artifacts/models/champion/metadata.json").read_text())
cols = meta["numeric"] + meta["categorical"]
rows = storage.read_parquet("data/reference/features.parquet").sample(3, random_state=1)


def clean(v):
    if pd.isna(v):
        return None
    return v if isinstance(v, str) else float(v)


payload = {"students": [
    {"id_student": int(r["id_student"]), "features": {c: clean(r[c]) for c in cols}}
    for _, r in rows.iterrows()
]}
json.dump(payload, sys.stdout, indent=2)
