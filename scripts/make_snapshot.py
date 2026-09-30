"""Copy the small set of pipeline outputs the dashboard reads into dashboard/snapshot/.

The snapshot lets the dashboard run on Streamlit Community Cloud, which cannot reach the S3
bucket. Run it after a pipeline replay, then commit dashboard/snapshot/."""
import shutil
from pathlib import Path

DATA = Path("data")
MODELS = Path("artifacts/models")
OUT = Path("dashboard/snapshot")

if OUT.exists():
    shutil.rmtree(OUT)

copies = [(DATA / "state" / "stream_state.json", OUT / "data" / "state" / "stream_state.json")]
for kind in ("processed", "predictions"):
    copies += [(p, OUT / "data" / kind / p.name) for p in sorted((DATA / kind).glob("*.parquet"))]
copies += [(p, OUT / "data" / "reports" / p.parent.name / p.name)
           for p in sorted((DATA / "reports").glob("*/drift.json"))]
for name in ("champion", "v1"):
    copies += [(p, OUT / "models" / name / p.name) for p in sorted((MODELS / name).glob("*"))]

for src, dst in copies:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(f"{len(copies)} files, {size / 1e6:.1f} MB -> {OUT}")
