"""Convert the raw OULAD CSVs into typed Parquet files (the bronze layer)."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

TABLES = {
    "courses": "courses.csv",
    "assessments": "assessments.csv",
    "vle": "vle.csv",
    "student_info": "studentInfo.csv",
    "registration": "studentRegistration.csv",
    "student_assessment": "studentAssessment.csv",
    "student_vle": "studentVle.csv",
}


def csv_to_parquet(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for name, filename in TABLES.items():
        src = raw_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}. Unzip OULAD into {raw_dir}.")
        dst = out_dir / f"{name}.parquet"
        # Paths come from our own config, not user input, so f-strings are safe here.
        con.execute(
            f"COPY (SELECT * FROM read_csv_auto('{src.as_posix()}', header=true, nullstr='?')) "
            f"TO '{dst.as_posix()}' (FORMAT parquet)"
        )
        rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{dst.as_posix()}')").fetchone()[0]
        print(f"{name:20s} {rows:>12,d} rows -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/bronze")
    args = parser.parse_args()
    csv_to_parquet(Path(args.raw), Path(args.out))
