"""Download trained nnQC weights from the Hugging Face Hub.

Checkpoints are too large for git, so they live in a Hub model repo
(default ``sanbast/nnQC``, private). ``download_weights`` fetches the four
files a task needs (autoencoder, diffusion UNet, cross-attention grid, slice
embedding) into ``trained_weights/<task>/`` so the rest of the package can
load them transparently.

    import nnqc
    nnqc.download_weights("prostate")            # -> trained_weights/prostate/
    nnqc.check("scan.nii.gz", "mask.nii.gz", task="prostate")

For a private repo you need a token with read access, supplied via the
``token`` argument, the ``HF_TOKEN`` environment variable, or a prior
``huggingface-cli login``.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_REPO_ID = "sanbast/nnQC"
# Files required to run inference / evaluation for a task.
WEIGHT_FILES = ("autoencoder.pt", "diffusion_unet.pt", "xa.pt", "embed.pt")


def _resolve_token(token):
    return token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def download_weights(
    task: str,
    repo_id: str = DEFAULT_REPO_ID,
    dest=None,
    token=None,
    files=WEIGHT_FILES,
    overwrite: bool = False,
) -> str:
    """Fetch a task's checkpoints from the Hub into ``dest`` (default
    ``trained_weights/<task>/``) and return that directory.

    Hub layout is ``weights/<task>/<file>.pt``.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to download weights. "
            "Install it with `uv pip install huggingface_hub` or `pip install nnqc[hub]`."
        ) from exc

    token = _resolve_token(token)
    dest = Path(dest) if dest is not None else Path("trained_weights") / task
    dest.mkdir(parents=True, exist_ok=True)

    for fname in files:
        target = dest / fname
        if target.exists() and not overwrite:
            print(f"[nnqc] {target} already present, skipping")
            continue
        cached = hf_hub_download(
            repo_id=repo_id, repo_type="model",
            filename=f"weights/{task}/{fname}", token=token,
        )
        shutil.copyfile(cached, target)
        print(f"[nnqc] fetched {fname} -> {target}")
    print(f"[nnqc] weights for '{task}' ready in {dest}")
    return str(dest)


def ensure_weights(task, model_dir, token=None, repo_id=DEFAULT_REPO_ID, files=WEIGHT_FILES) -> bool:
    """Download weights into ``model_dir`` if any required file is missing.

    Returns True if a download was triggered. Used by check/evaluate to
    transparently fetch weights when a known ``task`` preset is used.
    """
    model_dir = Path(model_dir)
    if all((model_dir / f).exists() for f in files):
        return False
    print(f"[nnqc] weights missing in {model_dir}; downloading '{task}' from {repo_id}...")
    download_weights(task, repo_id=repo_id, dest=model_dir, token=token, files=files)
    return True
