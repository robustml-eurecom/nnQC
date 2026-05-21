"""Configuration handling for nnQC.

The training internals consume a flat namespace (``cfg.model_dir``,
``cfg.num_classes``, ``cfg.autoencoder_train["lr"]`` ...). Historically this
namespace was built by flattening two JSON files (an ``env.json`` with paths /
dataset settings and a ``config.json`` with network + training hyper-params).

This module keeps those JSON files as the canonical templates but lets the
notebook API and CLI override any field with plain keyword arguments, e.g.::

    cfg = resolve_config(task="prostate", stage="diffusion",
                         overrides={"epochs": 4000, "lr": 2.5e-5,
                                    "scheduler": "cosine"})

Friendly keywords (``epochs``, ``lr``, ``batch_size`` ...) are routed into the
correct nested block automatically.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

_AE_BLOCK = "autoencoder_train"
_DIFF_BLOCK = "diffusion_train"
_NOISE_BLOCK = "NoiseScheduler"

# Friendly kwarg -> canonical top-level attribute.
_TOPLEVEL_ALIASES = {
    "data_dir": "data_base_dir",
    "resume": "resume_ckpt",
}
_TOPLEVEL_KEYS = {
    "data_base_dir", "model_dir", "tfevent_path", "output_dir",
    "task", "modality", "num_classes", "sample_axis", "channel",
    "is_msd", "image_pattern", "label_pattern", "recursive", "using_gt",
    "download", "resume_ckpt", "start_epoch",
    "spatial_dims", "image_channels", "latent_channels",
}
# Friendly kwarg -> key inside the *stage* training block (autoencoder/diffusion).
_BLOCK_COMMON = {
    "epochs": "max_epochs",
    "max_epochs": "max_epochs",
    "lr": "lr",
    "batch_size": "batch_size",
    "patch_size": "patch_size",
    "val_interval": "val_interval",
}
# Autoencoder-only block keys.
_AE_ONLY = {
    "perceptual_weight": "perceptual_weight",
    "kl_weight": "kl_weight",
    "recon_loss": "recon_loss",
}
# Diffusion-only block keys.
_DIFF_ONLY = {
    "warmup_dice_epochs": "warmup_dice_epochs",
    "scheduler": "scheduler",
    "warmup_epochs": "warmup_epochs",
    "lambda_recon": "lambda_recon",
    "ema_decay": "ema_decay",
    "weight_decay": "weight_decay",
}
# Noise-scheduler block keys.
_NOISE = {
    "num_train_timesteps": "num_train_timesteps",
    "beta_start": "beta_start",
    "beta_end": "beta_end",
    "clip_sample": "clip_sample",
}
_RAW_BLOCKS = {_AE_BLOCK, _DIFF_BLOCK, _NOISE_BLOCK, "autoencoder_def", "diffusion_def"}


def _load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _presets_root():
    return resources.files("nnqc") / "presets"


def available_tasks() -> list[str]:
    """Names of the task presets bundled with the package."""
    root = _presets_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _preset_paths(task: str):
    base = _presets_root() / task
    cfg, env = base / "config.json", base / "env.json"
    if not cfg.is_file() or not env.is_file():
        raise ValueError(
            f"Unknown task preset {task!r}. Available presets: {available_tasks()}. "
            "Otherwise pass explicit config=<path> and env=<path>."
        )
    return cfg, env


def _apply_overrides(ns: argparse.Namespace, overrides: Mapping[str, Any], stage: str) -> None:
    ae = dict(getattr(ns, _AE_BLOCK, {}) or {})
    diff = dict(getattr(ns, _DIFF_BLOCK, {}) or {})
    noise = dict(getattr(ns, _NOISE_BLOCK, {}) or {})
    stage_block = ae if stage == "autoencoder" else diff

    for key, val in overrides.items():
        if val is None:
            continue
        canon = _TOPLEVEL_ALIASES.get(key, key)
        if canon in _TOPLEVEL_KEYS:
            setattr(ns, canon, val)
        elif key in _BLOCK_COMMON:
            stage_block[_BLOCK_COMMON[key]] = val
        elif key in _AE_ONLY:
            ae[_AE_ONLY[key]] = val
        elif key in _DIFF_ONLY:
            diff[_DIFF_ONLY[key]] = val
        elif key in _NOISE:
            noise[_NOISE[key]] = val
        elif key in _RAW_BLOCKS and isinstance(val, Mapping):
            cur = dict(getattr(ns, key, {}) or {})
            cur.update(val)
            setattr(ns, key, cur)
            if key == _AE_BLOCK:
                ae = dict(cur)
            elif key == _DIFF_BLOCK:
                diff = dict(cur)
            elif key == _NOISE_BLOCK:
                noise = dict(cur)
        else:
            # Escape hatch: set any other key as a top-level attribute.
            setattr(ns, canon, val)

    setattr(ns, _AE_BLOCK, ae)
    setattr(ns, _DIFF_BLOCK, diff)
    setattr(ns, _NOISE_BLOCK, noise)


def resolve_config(
    config: str | Path | Mapping | None = None,
    env: str | Path | Mapping | None = None,
    task: str | None = None,
    *,
    stage: str,
    overrides: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    """Merge an ``env`` + ``config`` pair (or a named task preset) into one flat
    namespace and apply keyword overrides.

    Parameters
    ----------
    config, env
        Paths to JSON files (or already-loaded dicts). Required unless ``task``
        is given.
    task
        Name of a bundled preset (see :func:`available_tasks`). Used only when
        ``config`` and ``env`` are both omitted.
    stage
        ``"autoencoder"`` or ``"diffusion"`` - decides which training block the
        common overrides (``epochs``, ``lr`` ...) route into.
    overrides
        Mapping of friendly keyword overrides.
    """
    if stage not in ("autoencoder", "diffusion"):
        raise ValueError(f"stage must be 'autoencoder' or 'diffusion', got {stage!r}")

    if config is None and env is None:
        if task is None:
            raise ValueError(
                "Provide either task=<preset> or both config=<path> and env=<path>."
            )
        cfg_path, env_path = _preset_paths(task)
        config_dict = _load_json(cfg_path)
        env_dict = _load_json(env_path)
    else:
        if config is None or env is None:
            raise ValueError("When not using task=, pass both config= and env=.")
        config_dict = config if isinstance(config, Mapping) else _load_json(config)
        env_dict = env if isinstance(env, Mapping) else _load_json(env)

    merged: dict[str, Any] = {}
    merged.update(copy.deepcopy(dict(env_dict)))
    merged.update(copy.deepcopy(dict(config_dict)))
    ns = argparse.Namespace(**merged)

    if overrides:
        _apply_overrides(ns, dict(overrides), stage=stage)
    return ns
