"""The simulated stream: which batch comes next, and the pipeline's persistent state."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from oulad import storage
from oulad.config import Settings
from oulad.data import STREAM_TERMS


def state_uri(s: Settings) -> str:
    return storage.join(s.data_uri, "state", "stream_state.json")


def load_state(s: Settings) -> dict:
    return storage.read_json(state_uri(s))


def save_state(s: Settings, state: dict) -> None:
    storage.write_json(state, state_uri(s))


def batch_schedule(bronze_dir: Path) -> list[tuple[str, str]]:
    info = pd.read_parquet(
        Path(bronze_dir) / "student_info.parquet", columns=["code_module", "code_presentation"]
    )
    pairs = info[info["code_presentation"].isin(STREAM_TERMS)].drop_duplicates().copy()
    pairs["term_order"] = pairs["code_presentation"].map({t: i for i, t in enumerate(STREAM_TERMS)})
    pairs = pairs.sort_values(["term_order", "code_module"])
    return list(pairs[["code_module", "code_presentation"]].itertuples(index=False, name=None))
