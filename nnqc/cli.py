"""``nnqc`` command-line interface.

A single dispatcher with subcommands::

    nnqc train-autoencoder --task prostate --epochs 500 --lr 5e-5
    nnqc train-diffusion   --task prostate --epochs 4000 --scheduler cosine
    nnqc evaluate          --task prostate --num-steps 5
    nnqc list-tasks

Every subcommand accepts either ``--task <preset>`` or an explicit
``--config <path> --env <path>`` pair, plus overrides for any hyper-parameter.
"""
from __future__ import annotations

import argparse

from nnqc import __version__
from nnqc.config import available_tasks


def _add_common(p: argparse.ArgumentParser) -> None:
    src = p.add_argument_group("config source")
    src.add_argument("-t", "--task", default=None, help="bundled preset name (see `nnqc list-tasks`)")
    src.add_argument("-c", "--config", default=None, help="path to a config.json")
    src.add_argument("-e", "--env", default=None, help="path to an env.json")

    exe = p.add_argument_group("execution")
    exe.add_argument("-g", "--gpus", type=int, default=1, help="number of GPUs (>1 uses torchrun DDP)")
    exe.add_argument("--device", default=None, help="device for single-GPU runs, e.g. 0 or cuda:2")
    exe.add_argument("--seed", type=int, default=42)

    ov = p.add_argument_group("common overrides")
    ov.add_argument("--epochs", type=int, default=None)
    ov.add_argument("--lr", type=float, default=None)
    ov.add_argument("--batch-size", type=int, default=None)
    ov.add_argument("--patch-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    ov.add_argument("--val-interval", type=int, default=None)
    ov.add_argument("--data-dir", default=None)
    ov.add_argument("--model-dir", default=None)
    ov.add_argument("--tfevent-path", default=None)
    ov.add_argument("--output-dir", default=None)
    ov.add_argument("--num-classes", type=int, default=None)
    ov.add_argument("--modality", default=None)
    ov.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    ov.add_argument("--start-epoch", type=int, default=None)


# Argparse dest -> friendly override key (most are identical).
_OVERRIDE_KEYS = [
    "epochs", "lr", "batch_size", "patch_size", "val_interval",
    "data_dir", "model_dir", "tfevent_path", "output_dir",
    "num_classes", "modality", "resume", "start_epoch",
    # autoencoder-only
    "perceptual_weight", "kl_weight", "recon_loss",
    # diffusion-only
    "warmup_dice_epochs", "scheduler", "warmup_epochs", "lambda_recon", "ema_decay",
    "num_train_timesteps", "beta_start", "beta_end",
]


def _collect_overrides(args) -> dict:
    out = {}
    for key in _OVERRIDE_KEYS:
        if hasattr(args, key):
            val = getattr(args, key)
            if val is not None:
                out[key] = list(val) if key == "patch_size" else val
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nnqc", description="Segmentation QC via latent diffusion.")
    parser.add_argument("--version", action="version", version=f"nnqc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ae = sub.add_parser("train-autoencoder", help="train the AutoencoderKL")
    _add_common(p_ae)
    ae = p_ae.add_argument_group("autoencoder overrides")
    ae.add_argument("--perceptual-weight", type=float, default=None)
    ae.add_argument("--kl-weight", type=float, default=None)
    ae.add_argument("--recon-loss", choices=["l1", "l2", "dice_ce"], default=None)

    p_df = sub.add_parser("train-diffusion", help="train the diffusion UNet")
    _add_common(p_df)
    df = p_df.add_argument_group("diffusion overrides")
    df.add_argument("--warmup-dice-epochs", type=int, default=None)
    df.add_argument("--scheduler", choices=["cosine", "constant", "step", "exponential"], default=None)
    df.add_argument("--warmup-epochs", type=int, default=None)
    df.add_argument("--lambda-recon", type=float, default=None)
    df.add_argument("--ema-decay", type=float, default=None)
    df.add_argument("--num-train-timesteps", type=int, default=None)
    df.add_argument("--beta-start", type=float, default=None)
    df.add_argument("--beta-end", type=float, default=None)

    p_ev = sub.add_parser("evaluate", help="sample reconstructions and write panels")
    _add_common(p_ev)
    ev = p_ev.add_argument_group("evaluate options")
    ev.add_argument("--checkpoint", choices=["last", "best"], default="last")
    ev.add_argument("--num-volumes", type=int, default=3)
    ev.add_argument("--num-steps", type=int, default=5)
    ev.add_argument("--step", type=int, default=0)

    sub.add_parser("list-tasks", help="list bundled task presets")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list-tasks":
        tasks = available_tasks()
        print("Available task presets:" if tasks else "No bundled task presets found.")
        for t in tasks:
            print(f"  - {t}")
        return 0

    # Import here so `nnqc list-tasks` / `--version` stay fast (no torch import).
    from nnqc.train import train_autoencoder, train_diffusion
    from nnqc.evaluate import evaluate

    overrides = _collect_overrides(args)
    common = dict(config=args.config, env=args.env, task=args.task, device=args.device, seed=args.seed)

    if args.command == "train-autoencoder":
        train_autoencoder(gpus=args.gpus, **common, **overrides)
    elif args.command == "train-diffusion":
        train_diffusion(gpus=args.gpus, **common, **overrides)
    elif args.command == "evaluate":
        evaluate(
            checkpoint=args.checkpoint, num_volumes=args.num_volumes,
            num_steps=args.num_steps, step=args.step, **common, **overrides,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
