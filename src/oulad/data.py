"""Load bronze tables and define the prediction population and label."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

KEYS = ["code_module", "code_presentation", "id_student"]
PRESENTATION = ["code_module", "code_presentation"]
AT_RISK_RESULTS = ("Fail", "Withdrawn")
SMALL_TABLES = ["courses", "assessments", "vle", "student_info", "registration", "student_assessment"]


TRAIN_TERMS = ("2013B", "2013J")    # used to train the first model
STREAM_TERMS = ("2014B", "2014J")   # replayed later as the live stream


def local_bronze(data_uri: str, cache_dir: str = "/tmp/oulad/bronze") -> Path:
    """Local folder with the bronze Parquet files. Downloads them first when data_uri is on S3."""
    from oulad import storage

    return storage.materialize_dir(storage.join(data_uri, "bronze"), cache_dir)


def load_tables(bronze_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every table except student_vle, which is aggregated in DuckDB instead."""
    return {name: pd.read_parquet(Path(bronze_dir) / f"{name}.parquet") for name in SMALL_TABLES}


def build_population(
    tables: dict[str, pd.DataFrame],
    cutoff_day: int,
    presentations: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Students still enrolled at the cutoff, with the binary at_risk label."""
    info = tables["student_info"].merge(tables["registration"], on=KEYS, how="left")
    if presentations is not None:
        wanted = pd.DataFrame(presentations, columns=PRESENTATION)
        info = info.merge(wanted, on=PRESENTATION, how="inner")
    still_enrolled = info["date_unregistration"].isna() | (info["date_unregistration"] > cutoff_day)
    pop = info.loc[still_enrolled].copy()
    pop["at_risk"] = pop["final_result"].isin(AT_RISK_RESULTS).astype(int)
    return pop.reset_index(drop=True)
