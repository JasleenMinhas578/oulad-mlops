"""Read and write files on local disk or S3 through one small interface."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd


def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def join(base: str, *parts: str) -> str:
    return "/".join([base.rstrip("/"), *[p.strip("/") for p in parts]])


def _split(uri: str) -> tuple[str, str]:
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def _s3():
    import boto3  # imported lazily so local runs never need AWS credentials

    return boto3.client("s3")


def read_bytes(uri: str) -> bytes:
    if is_s3(uri):
        bucket, key = _split(uri)
        return _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(uri).read_bytes()


def write_bytes(uri: str, data: bytes) -> None:
    if is_s3(uri):
        bucket, key = _split(uri)
        _s3().put_object(Bucket=bucket, Key=key, Body=data)
        return
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def exists(uri: str) -> bool:
    if is_s3(uri):
        from botocore.exceptions import ClientError

        bucket, key = _split(uri)
        try:
            _s3().head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
    return Path(uri).exists()


def read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def write_json(obj: dict, uri: str) -> None:
    write_bytes(uri, json.dumps(obj, indent=2, default=str).encode())


def read_parquet(uri: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(read_bytes(uri)))


def write_parquet(df: pd.DataFrame, uri: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    write_bytes(uri, buf.getvalue())


def list_files(uri: str) -> list[str]:
    """Every file under a prefix, as full URIs."""
    if is_s3(uri):
        bucket, key = _split(uri)
        prefix = key.rstrip("/") + "/"
        out: list[str] = []
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            out += [f"s3://{bucket}/{obj['Key']}" for obj in page.get("Contents", [])]
        return out
    root = Path(uri)
    return [str(p) for p in root.rglob("*") if p.is_file()] if root.exists() else []


def copy_dir(src: str, dst: str) -> None:
    """Copy every file under src to dst. Works local->local, local->S3, S3->S3, S3->local."""
    base = src.rstrip("/")
    for f in list_files(src):
        write_bytes(join(dst, f[len(base) + 1:]), read_bytes(f))


def materialize_dir(uri: str, local_dir: str | Path) -> Path:
    """Return a local directory with the contents of uri (downloads when uri is on S3)."""
    if not is_s3(uri):
        return Path(uri)
    local_dir = Path(local_dir)
    copy_dir(uri, str(local_dir))
    return local_dir
