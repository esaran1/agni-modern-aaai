from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_manifest(
    dataset_path: Path,
    config_dict: dict[str, Any],
    row_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": hash_file(dataset_path),
        "config_sha256": sha256_bytes(json.dumps(config_dict, sort_keys=True).encode("utf-8")),
        "row_count": row_count,
    }
    if extra:
        manifest.update(extra)
    return manifest
