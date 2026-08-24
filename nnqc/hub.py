"""Download trained nnQC weights from Zenodo.

Checkpoints are too large for git, so they are published as Zenodo records, one
deposition per task. ``download_weights`` fetches the four files a task needs
(autoencoder, diffusion UNet, cross-attention grid, slice embedding) into
``trained_weights/<task>/`` so the rest of the package can load them
transparently.

    import nnqc
    nnqc.download_weights("prostate")            # -> trained_weights/prostate/
    nnqc.check("scan.nii.gz", "mask.nii.gz", task="prostate")

The record id for a task is resolved from, in order of precedence:

1. the ``record_id`` argument,
2. the ``NNQC_ZENODO_RECORD_<TASK>`` environment variable (task upper-cased,
   dashes become underscores, e.g. ``NNQC_ZENODO_RECORD_PROSTATE``),
3. the built-in :data:`ZENODO_RECORDS` map.

Zenodo is public, so no token is needed. Only the Python standard library is
used (``urllib``), so downloading weights pulls in no extra dependency.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records"

# Zenodo record id per task. Filled in as depositions are published; until
# then, pass record_id= / --record or set NNQC_ZENODO_RECORD_<TASK>.
ZENODO_RECORDS: dict[str, str] = {}

# Files required to run inference / evaluation for a task.
WEIGHT_FILES = ("autoencoder.pt", "diffusion_unet.pt", "xa.pt", "embed.pt")


def _resolve_record_id(task: str, record_id=None) -> str:
    rid = (
        record_id
        or os.getenv(f"NNQC_ZENODO_RECORD_{task.upper().replace('-', '_')}")
        or ZENODO_RECORDS.get(task)
    )
    if not rid:
        raise RuntimeError(
            f"No Zenodo record id known for task {task!r}. Pass it explicitly "
            f"(`nnqc download {task} --record <id>` or download_weights(record_id=...)), "
            f"or set the NNQC_ZENODO_RECORD_{task.upper().replace('-', '_')} environment "
            "variable. Once the deposition is published, the id is also added to "
            "nnqc.hub.ZENODO_RECORDS."
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


def download_weights(
    task: str,
    record_id=None,
    dest=None,
    files=WEIGHT_FILES,
    overwrite: bool = False,
) -> str:
    """Fetch a task's checkpoints from Zenodo into ``dest`` (default
    ``trained_weights/<task>/``) and return that directory.

    The record id is resolved as described in the module docstring. The
    deposition is expected to contain one file per entry in ``files``
    (e.g. ``autoencoder.pt``), not an archive.
    """
    record_id = _resolve_record_id(task, record_id)
    dest = Path(dest) if dest is not None else Path("trained_weights") / task
    dest.mkdir(parents=True, exist_ok=True)

    record = _fetch_record(record_id)
    links = {}
    for f in record.get("files", []):
        url = (f.get("links") or {}).get("self")
        if url:
            links[f.get("key", "")] = url

    for fname in files:
        target = dest / fname
        if target.exists() and not overwrite:
            print(f"[nnqc] {target} already present, skipping")
            continue
        if fname not in links:
            raise RuntimeError(
                f"Zenodo record {record_id} has no file {fname!r} "
                f"(it contains: {sorted(links)}). Check the record id for task {task!r}."
            )
        print(f"[nnqc] downloading {fname} from Zenodo record {record_id}...")
        _download(links[fname], target)
        print(f"[nnqc] fetched {fname} -> {target}")
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
