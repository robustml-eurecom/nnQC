"""Download trained nnQC weights from Zenodo.

Checkpoints are too large for git, so they are published on Zenodo as a single
record shipping one archive, ``nnQC_pretrained_weights.zip``, which contains one
folder per task::

    nnQC_pretrained_weights.zip
    ├── weight_liver/{autoencoder.pt, diffusion_unet.pt, xa.pt, embed.pt, scale_factor.txt}
    ├── weight_prostate/...
    ├── weight_cardiac/...
    └── weight_spleen/...

``download_weights("prostate")`` downloads the archive once (it is cached under
``trained_weights/.cache/``) and extracts only the requested task into
``trained_weights/<task>/`` so the rest of the package can load it
transparently.

    import nnqc
    nnqc.download_weights("prostate")            # -> trained_weights/prostate/
    nnqc.check("scan.nii.gz", "mask.nii.gz", task="prostate")

The record id is resolved from, in order of precedence:

1. the ``record_id`` argument,
2. the ``NNQC_ZENODO_RECORD_<TASK>`` environment variable (task upper-cased,
   dashes become underscores, e.g. ``NNQC_ZENODO_RECORD_PROSTATE``),
3. the ``NNQC_ZENODO_RECORD`` environment variable,
4. the built-in :data:`ZENODO_RECORDS` map, then :data:`ZENODO_RECORD`.

Zenodo is public, so no token is needed. Only the Python standard library is
used (``urllib``, ``zipfile``), so downloading weights pulls in no extra
dependency.
"""
from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records"

# Name of the archive inside the Zenodo record, and the per-task folder prefix
# used within it.
ARCHIVE_NAME = "nnQC_pretrained_weights.zip"
ZIP_PREFIX = "weight_"

# Zenodo record id of the shared weights archive (per-task overrides allowed in
# ZENODO_RECORDS). Filled in once the deposition is published; until then, pass
# record_id= / --record or set NNQC_ZENODO_RECORD.
ZENODO_RECORD = ""
ZENODO_RECORDS: dict[str, str] = {}

# Files required to run inference / evaluation for a task.
WEIGHT_FILES = ("autoencoder.pt", "diffusion_unet.pt", "xa.pt", "embed.pt", "scale_factor.txt")


def _resolve_record_id(task: str, record_id=None) -> str:
    rid = (
        record_id
        or os.getenv(f"NNQC_ZENODO_RECORD_{task.upper().replace('-', '_')}")
        or os.getenv("NNQC_ZENODO_RECORD")
        or ZENODO_RECORDS.get(task)
        or ZENODO_RECORD
    )
    if not rid:
        raise RuntimeError(
            f"No Zenodo record id known for task {task!r}. Pass it explicitly "
            f"(`nnqc download {task} --record <id>` or download_weights(record_id=...)), "
            "or set the NNQC_ZENODO_RECORD environment variable. Once the deposition "
            "is published, the id is also added to nnqc.hub.ZENODO_RECORD."
        )
    return str(rid)


def _fetch_record(record_id: str) -> dict:
    url = f"{ZENODO_API}/{record_id}"
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (fixed https host)
        return json.loads(resp.read().decode())


def _download(url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r[nnqc]   {target.name}: {got / 1e6:.0f}/{total / 1e6:.0f} MB",
                      end="", flush=True)
        if total:
            print()
    tmp.replace(target)


def _archive_url(record: dict, record_id: str) -> str:
    for f in record.get("files", []):
        if f.get("key") == ARCHIVE_NAME and (f.get("links") or {}).get("self"):
            return f["links"]["self"]
    raise RuntimeError(
        f"Zenodo record {record_id} has no file {ARCHIVE_NAME!r} "
        f"(it contains: {[f.get('key') for f in record.get('files', [])]})."
    )


def _cached_archive(record: dict, record_id: str, cache_dir: Path, overwrite: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / ARCHIVE_NAME
    if archive.exists() and not overwrite:
        return archive
    url = _archive_url(record, record_id)
    print(f"[nnqc] downloading {ARCHIVE_NAME} from Zenodo record {record_id}...")
    _download(url, archive)
    return archive


def _extract_task(archive: Path, task: str, dest: Path, files, overwrite: bool) -> None:
    """Extract ``weight_<task>/<file>`` members of the archive into ``dest``.

    Tolerates an optional top-level folder inside the zip (e.g.
    ``nnQC_pretrained_weights/weight_liver/autoencoder.pt``).
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for fname in files:
            target = dest / fname
            if target.exists() and not overwrite:
                print(f"[nnqc] {target} already present, skipping")
                continue
            member = next(
                (n for n in names
                 if Path(n).name == fname and f"{ZIP_PREFIX}{task}" in Path(n).parts),
                None,
            )
            if member is None:
                raise RuntimeError(
                    f"{archive.name} has no member '{ZIP_PREFIX}{task}/{fname}' "
                    f"(task folders found: "
                    f"{sorted({p for n in names for p in Path(n).parts if p.startswith(ZIP_PREFIX)})}). "
                    f"Check the archive layout for task {task!r}."
                )
            with zf.open(member) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            print(f"[nnqc] extracted {member} -> {target}")


def download_weights(
    task: str,
    record_id=None,
    dest=None,
    files=WEIGHT_FILES,
    overwrite: bool = False,
) -> str:
    """Fetch a task's checkpoints from Zenodo into ``dest`` (default
    ``trained_weights/<task>/``) and return that directory.

    The record id is resolved as described in the module docstring. The record
    must contain :data:`ARCHIVE_NAME` with a ``weight_<task>/`` folder per task.
    The archive is cached under ``trained_weights/.cache/`` so downloading a
    second task does not fetch it again.
    """
    record_id = _resolve_record_id(task, record_id)
    dest = Path(dest) if dest is not None else Path("trained_weights") / task
    dest.mkdir(parents=True, exist_ok=True)
    cache_dir = dest.parent / ".cache" if dest.parent.name == "trained_weights" else dest / ".cache"

    record = _fetch_record(record_id)
    archive = _cached_archive(record, record_id, cache_dir, overwrite)
    _extract_task(archive, task, dest, files, overwrite)
    print(f"[nnqc] weights for '{task}' ready in {dest}")
    return str(dest)


def ensure_weights(task, model_dir, record_id=None, files=WEIGHT_FILES) -> bool:
    """Download weights into ``model_dir`` if any required file is missing.

    Returns True if a download was triggered. Used by check/evaluate to
    transparently fetch weights when a known ``task`` preset is used.
    """
    model_dir = Path(model_dir)
    if all((model_dir / f).exists() for f in files):
        return False
    print(f"[nnqc] weights missing in {model_dir}; downloading '{task}' from Zenodo...")
    download_weights(task, record_id=record_id, dest=model_dir, files=files)
    return True
