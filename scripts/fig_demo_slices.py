#!/usr/bin/env python
"""Project-page interactive demo: one best-calibrated slice per organ.

Replaces the static strip panels (scripts/fig_demo_panels.py). For each organ
we reproduce one demo case - real TotalSegmentator output for liver, a
mid-severity synthetic corruption for the others - run the benchmark's
sampling path, and render ONE full-FOV axial slice in two views
(`s00_input.png`: scan + per-class input-mask contours, `s00_pgt.png`: scan +
per-class pGT contours). The slice shown is the one minimizing
|slice QC score - slice true Dice| (tie-broken by pGT-vs-GT Dice); the true
Dice is used for selection only and never appears on the page. Slices are
rotated 90 deg counterclockwise (np.rot90, scan and mask together) and
downscaled to ~384 px wide if the full field of view is larger.

    python scripts/fig_demo_slices.py --out-dir docs/assets
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

CLASS_COLORS = {1: (0.95, 0.55, 0.20), 2: (0.30, 0.70, 1.00), 3: (0.40, 0.90, 0.50)}
PGT_COLOR = (0.45, 0.95, 0.75)

UNIFIED = Path("/lustre/fsn1/projects/rech/rpv/uxl51vf/grace-med-data/unified_perclass")
UP_CACHE = Path(os.environ.get(
    "NNQC_UP_CACHE", "/lustre/fsn1/projects/rech/rpv/uxl51vf/nnqc/tmp/real_eval_up"))

CASES = {
    # Real TotalSegmentator prediction (true DSC ~0.88 vs native label,
    # results/real_segmentations_liver.json) - decent but visibly imperfect.
    "liver": dict(
        cfgdir="liver", steps=4, mode="real",
        image=str(UNIFIED / "images/test/Task03_Liver_liver_107_liver.nii.gz"),
        label=str(UNIFIED / "labels/test/Task03_Liver_liver_107_liver.nii.gz"),
        mask=str(UP_CACHE / "liver/Task03_Liver_liver_107/totalseg.nii.gz"),
        source="TotalSegmentator prediction", label_text="Liver", modality="CT · MSD Task03"),
    # Mid-severity synthetic corruptions of val cases with good pGT (results/).
    "prostate": dict(
        cfgdir="prostate", steps=4, mode="synthetic", split="all",
        volume="prostate_21_img.nii.gz", severity=0.2,   # mild corruption
        source="synthetic corruption (severity 0.2)",
        label_text="Prostate", modality="MRI · MSD Task05"),
    "cardiac": dict(
        cfgdir="cardiac_onehot", steps=5, mode="synthetic",
        volume="patient150_frame01_img.nii.gz", severity=0.4,  # recon 0.816 @ cardiac_onehot_1315052.json
        source="synthetic corruption (severity 0.4)",
        label_text="Cardiac", modality="MRI · ACDC"),
    "spleen": dict(
        cfgdir="spleen_ext", steps=5, samples=5, fill_holes=True, mode="synthetic", split="all",
        volume="spleen_9_img.nii.gz", severity=0.5,      # ext chain, ensemble preset
        pick="recon",  # show the slice where pGT is closest to GT
        source="synthetic corruption (severity 0.5)",
        label_text="Spleen", modality="CT · MSD Task09"),
}

DPI = 96
MAX_W = 384  # downscale the saved PNG to this width if the full FOV is larger


def save_view(path, img, mask, color_map):
    """One axial PNG: full-FOV grayscale scan + one contour per label id."""
    import io

    from PIL import Image

    h, w = img.shape
    fig = plt.figure(figsize=(w / 96, h / 96), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", vmin=float(np.percentile(img, 1)),
              vmax=float(np.percentile(img, 99)))
    if isinstance(color_map, dict):
        for c, rgb in color_map.items():
            m = mask == c
            if m.any():
                ax.contour(m, levels=[0.5], colors=[rgb], linewidths=1.2)
    else:
        m = mask > 0
        if m.any():
            ax.contour(m, levels=[0.5], colors=[color_map], linewidths=1.2)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    im.save(path)


def run_case(organ, case, out_dir, device="cuda", batch=16, seed=42):
    from monai.networks.schedulers import DDIMScheduler
    from monai.utils import set_determinism

    from nnqc.config import resolve_config
    from nnqc.refine import PostprocessConfig, postprocess_volume, sample_probabilities
    from nnqc.utils import define_instance, load_scale_factor, resolve_torch_device
    from nnqc.xa import CLIPCrossAttentionGrid
    from scripts.benchmark import corrupt_volume, multiclass_dice, preprocess, val_volumes

    root = os.environ.get("NNQC_ROOT", ".")
    cfg = resolve_config(f"{root}/configs/jz/{case['cfgdir']}/config.json",
                         f"{root}/configs/jz/{case['cfgdir']}/env.json",
                         None, stage="diffusion")
    set_determinism(seed)
    dev = resolve_torch_device(device)
    nc, md = cfg.num_classes, cfg.model_dir

    ae = define_instance(cfg, "autoencoder_def").to(dev).eval()
    ae.load_state_dict(torch.load(f"{md}/autoencoder.pt", map_location=dev, weights_only=True))
    unet = define_instance(cfg, "diffusion_def").to(dev).eval()
    unet.load_state_dict(torch.load(f"{md}/diffusion_unet.pt", map_location=dev, weights_only=True))
    _xa_sd = torch.load(f"{md}/xa.pt", map_location=dev, weights_only=True)
    xa = CLIPCrossAttentionGrid(output_dim=cfg.diffusion_def["cross_attention_dim"],
                                grid_reduction="column_softmax",
                                mask_gate=any(k.startswith("mask_state.") for k in _xa_sd)
                                ).to(dev).eval()
    xa.load_state_dict(_xa_sd)
    embed = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.GELU(),
        torch.nn.Linear(32, cfg.diffusion_def["cross_attention_dim"])).to(dev).eval()
    embed.load_state_dict(torch.load(f"{md}/embed.pt", map_location=dev, weights_only=True))
    sf = load_scale_factor(md, fallback=None)
    sched = DDIMScheduler(num_train_timesteps=cfg.NoiseScheduler["num_train_timesteps"],
                          schedule="scaled_linear_beta",
                          beta_start=cfg.NoiseScheduler["beta_start"],
                          beta_end=cfg.NoiseScheduler["beta_end"],
                          clip_sample=cfg.NoiseScheduler["clip_sample"])

    if case["mode"] == "real":
        img_path, lab_path = case["image"], case["label"]
        print(f"[demo] {organ}: real input {case['mask']}")
    else:
        vols = val_volumes(cfg, split=case.get("split", "val"))
        hits = [(i, p) for i, p in enumerate(vols)
                if Path(p["image"]).name == case["volume"]]
        if not hits:
            raise SystemExit(f"{organ}: {case['volume']} not in the val split")
        vi, pair = hits[0]
        img_path, lab_path = pair["image"], pair["label"]
        print(f"[demo] {organ}: {case['volume']} (val idx {vi}, seed {seed + vi}, "
              f"sev {case['severity']}, steps {case['steps']})")

    if case["mode"] == "real":
        # The upsampled TS mask is on the native grid. Preprocess image+mask
        # together so the scan and the candidate get the *same* z-crop (the
        # crop is driven by the label, so a separate image+GT run would land
        # on a different depth). GT is realigned afterwards and used ONLY to
        # pick the demo slice - it is never rendered or shown on the page.
        scan_v, cand_v = preprocess(img_path, case["mask"], cfg)
        scans = scan_v.permute(3, 0, 1, 2).contiguous().to(dev)   # [D,1,H,W]
        cand = cand_v.permute(3, 0, 1, 2).contiguous().to(dev)
        _, gt_v = preprocess(img_path, lab_path, cfg)
        gt2 = gt_v.permute(3, 0, 1, 2)[:, 0].numpy()              # [D2,H,W]
        # Same image, same affine: the only difference between the two runs is
        # the z SpatialCrop start, so a constant z offset realigns them.
        import nibabel as nib

        def zstart(p):
            a = np.asarray(nib.load(p).dataobj) > 0
            nz = np.nonzero(a.sum(axis=(0, 1)) > 0)[0]
            return int(nz[0]) if nz.size else 0
        off = zstart(case["mask"]) - zstart(lab_path)
        D = cand.shape[0]
        gt_np = np.zeros((gt2.shape[1], gt2.shape[2], D), dtype=gt2.dtype)
        for z in range(D):
            zg = z + off
            if 0 <= zg < gt2.shape[0]:
                gt_np[:, :, z] = gt2[zg]
    else:
        scan_v, gt_v = preprocess(img_path, lab_path, cfg)
        scans = scan_v.permute(3, 0, 1, 2).contiguous().to(dev)
        gts = gt_v.permute(3, 0, 1, 2).contiguous().to(dev)
        cand = corrupt_volume(gts.cpu(), nc, case["severity"], seed=seed + vi).to(dev)
        gt_np = gts[:, 0].cpu().numpy().transpose(1, 2, 0)        # [H,W,D]
    D = scans.shape[0]
    ratios = (torch.arange(D, device=dev).float() / max(D - 1, 1)).unsqueeze(1)

    probs = []
    for s in range(0, D, batch):
        e = min(s + batch, D)
        with torch.no_grad():
            if nc > 1:
                oh = F.one_hot(cand[s:e, 0].long().clamp(0, nc - 1),
                               nc).permute(0, 3, 1, 2).float()
            else:
                oh = (cand[s:e] > 0.5).float()
            lat = ae.encode_stage_2_inputs(oh).shape
            p = sample_probabilities(autoencoder=ae, unet=unet, xa=xa, embed=embed,
                                     scheduler=sched, scale_factor=sf,
                                     scans=scans[s:e], labels=cand[s:e],
                                     ratios=ratios[s:e], num_classes=nc,
                                     latent_shape=lat, num_steps=case["steps"],
                                     num_samples=case.get("samples", 1),
                                     seed=seed, device=dev)
        probs.append(p.float().cpu())
    prob = torch.cat(probs, 0).numpy().transpose(1, 2, 3, 0)  # [C,H,W,D]
    recon = postprocess_volume(prob, nc,
                               PostprocessConfig(fill_holes=bool(case.get("fill_holes"))))

    cand_np = cand[:, 0].cpu().numpy().transpose(1, 2, 0)     # [H,W,D]
    imgs = scans[:, 0].cpu().numpy().transpose(1, 2, 0)
    qc = multiclass_dice(cand_np, recon, nc)
    print(f"[demo] {organ}: qc={qc:.4f}")

    # ---- pick the single best-calibrated slice -----------------------------
    # min |slice QC - slice true Dice| (MAE criterion), tie-broken by pGT-vs-GT
    # Dice so the shown slice also reconstructs well. true Dice is selection-
    # only; it never leaves this script. Require BOTH candidate and GT to be
    # non-empty on the slice: an empty-input slice (qc = true = 0) is perfectly
    # calibrated but shows nothing on the input side.
    best = None
    for z in range(D):
        c_z, r_z, g_z = cand_np[..., z], recon[..., z], gt_np[..., z]
        if not ((c_z > 0).any() and (g_z > 0).any()):
            continue
        q_z = multiclass_dice(c_z, r_z, nc)
        t_z = multiclass_dice(c_z, g_z, nc)
        rg_z = multiclass_dice(r_z, g_z, nc)
        if case.get("pick") == "recon":
            key = (-rg_z, abs(q_z - t_z))
        else:
            key = (abs(q_z - t_z), -rg_z)
        if best is None or key < best[0]:
            best = (key, z, q_z, t_z, rg_z)
    _, z, slice_qc, slice_true, slice_rg = best
    print(f"[demo] {organ}: slice {z}/{D}  slice_qc={slice_qc:.4f} "
          f"slice_true={slice_true:.4f}  pgt_gt={slice_rg:.4f} "
          f"(mae={abs(slice_qc - slice_true):.4f})")

    cmap_in = CLASS_COLORS if nc > 1 else CLASS_COLORS[1]
    cmap_pgt = CLASS_COLORS if nc > 1 else PGT_COLOR
    odir = Path(out_dir) / f"demo_{organ}"
    odir.mkdir(parents=True, exist_ok=True)
    for old in odir.glob("*.png"):
        old.unlink()
    img = np.rot90(imgs[..., z])
    save_view(odir / "s00_input.png", img, np.rot90(cand_np[..., z]), cmap_in)
    save_view(odir / "s00_pgt.png", img, np.rot90(recon[..., z]), cmap_pgt)
    print(f"[demo] {organ}: wrote slice pair to {odir}")

    return dict(label=case["label_text"], modality=case["modality"],
                source=case["source"], steps=case["steps"], slice=int(z),
                qc_score=round(float(qc), 3),
                slice_qc_score=round(float(slice_qc), 3),
                dir=f"assets/demo_{organ}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organs", nargs="+", default=list(CASES))
    ap.add_argument("--out-dir", default="docs/assets")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    scores = {}
    for organ in args.organs:
        scores[organ] = run_case(organ, CASES[organ], args.out_dir, device=args.device)
        torch.cuda.empty_cache()
    out = Path(args.out_dir) / "demo_scores.json"
    out.write_text(json.dumps(scores, indent=2))
    print(f"[demo] wrote {out}")


if __name__ == "__main__":
    main()
