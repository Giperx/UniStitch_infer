"""Dataset-level HM metrics for pairwise stitches.

Scores the full canvas and the left, center, and right thirds. The mask is
the sparse GT mask only. L+R and L+R+C are means of those region means.
Predictions are the GT-sized crops under outputs/<dataset>/<scene>/pairwise/crop.
"""
import argparse
import os
from datetime import datetime

import numpy as np
from PIL import Image

from eval_sparse_full import find_gt, frame_hm, match_gt
from eval_sparse_regions import REGION_ORDER, region_slices

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = "/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub/datasets"
SPECS = {
    "nuscenes": {
        "pred": os.path.join(ROOT, "outputs/nuscenes"),
        "gt": os.path.join(WORK, "nuscenes/sparseWideFOVImages3_1554x294"),
        "cam": 5,
    },
    "lyft1920": {
        "pred": os.path.join(ROOT, "outputs/lyft1920"),
        "gt": os.path.join(WORK, "lyft/1920_sparseWideFOVImages3_1554x294"),
        "cam": 5,
    },
    "lyft1224": {
        "pred": os.path.join(ROOT, "outputs/lyft1224"),
        "gt": os.path.join(WORK, "lyft/1224_sparseWideFOVImages3_1554x434"),
        "cam": 5,
    },
    "widedrive3": {
        "pred": os.path.join(ROOT, "outputs/widedrive3"),
        "gt": os.path.join(WORK, "WideDrive_processed/sparseWideFOVImages3_1554x294"),
        "cam": 2,
    },
}


def mean_rows(rows):
    if not rows:
        return None
    return {
        "hm_psnr": float(np.mean([row["hm_psnr"] for row in rows if np.isfinite(row["hm_psnr"])])),
        "hm_ssim": float(np.mean([row["hm_ssim"] for row in rows])),
        "hm_mae": float(np.mean([row["hm_mae"] for row in rows])),
        "hm_rmse": float(np.mean([row["hm_rmse"] for row in rows])),
        "n": len(rows),
    }


def mean_of(stats):
    keys = ("hm_psnr", "hm_ssim", "hm_mae", "hm_rmse")
    return {key: float(np.mean([item[key] for item in stats])) for key in keys}


def format_stats(stats, frames=True):
    text = (
        f"HM-PSNR: {stats['hm_psnr']:.2f}, HM-SSIM: {stats['hm_ssim']:.4f}, "
        f"HM-MAE: {stats['hm_mae'] * 255:.2f}, HM-RMSE: {stats['hm_rmse'] * 255:.2f}"
    )
    if frames and "n" in stats:
        text += f", Frames: {stats['n']}"
    return text


def scene_crops(pred_root):
    scenes = []
    if not os.path.isdir(pred_root):
        return scenes
    for name in sorted(os.listdir(pred_root)):
        crop = os.path.join(pred_root, name, "pairwise", "crop")
        if os.path.isdir(crop):
            scenes.append((name, crop))
    return scenes


def score_spec(spec):
    full_rows = []
    region_rows = {name: [] for name in REGION_ORDER}
    per_scene = []
    skipped = 0
    for scene, pred_dir in scene_crops(spec["pred"]):
        gt_dir = os.path.join(spec["gt"], scene)
        names = sorted(name for name in os.listdir(pred_dir) if name.lower().endswith(".png"))
        scene_full = []
        scene_regions = {name: [] for name in REGION_ORDER}
        for name in names:
            frame = os.path.splitext(name)[0]
            gt_path, mask_path = find_gt(gt_dir, frame, spec["cam"])
            if gt_path is None:
                skipped += 1
                continue
            render = np.array(Image.open(os.path.join(pred_dir, name)).convert("RGB")).astype(np.float64) / 255.0
            gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.float64) / 255.0
            mask = np.array(Image.open(mask_path).convert("L")) > 127
            gt, mask = match_gt(render, gt, mask)
            row = frame_hm(render, gt, mask)
            if row is None:
                skipped += 1
                continue
            scene_full.append(row)
            slices = region_slices(render.shape[1])
            for region, (x0, x1) in slices.items():
                region_row = frame_hm(render[:, x0:x1], gt[:, x0:x1], mask[:, x0:x1])
                if region_row is not None:
                    scene_regions[region].append(region_row)
        if scene_full:
            full_rows.extend(scene_full)
            for region in REGION_ORDER:
                region_rows[region].extend(scene_regions[region])
            per_scene.append((
                scene,
                mean_rows(scene_full),
                {name: mean_rows(rows) for name, rows in scene_regions.items()},
            ))
            print(f"  {scene}: {len(scene_full)} frames", flush=True)
    return full_rows, region_rows, per_scene, skipped


def report_lines(title, full_rows, region_rows, per_scene, skipped, extra):
    full_stats = mean_rows(full_rows)
    region_stats = {name: mean_rows(region_rows[name]) for name in REGION_ORDER}
    lines = [
        f"=== {title} pairwise HM ===",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        *extra,
        "Mask: sparse GT mask only. LPIPS is not computed.",
        "HM-SSIM is the per-pixel luminance term.",
        "L+R and L+R+C are unweighted means of the region means.",
        "MAE and RMSE are on the 0-255 scale. PSNR uses [0, 1] images.",
        f"Skipped frames: {skipped}.",
        "",
        "Full",
        format_stats(full_stats) if full_stats else "no frames",
        "",
        "Regions",
    ]
    for name in REGION_ORDER:
        stats = region_stats[name]
        lines.append(f"{name:8s}: {format_stats(stats) if stats else 'no frames'}")
    if all(region_stats[name] for name in ("Left", "Right")):
        lines.append("L+R    : " + format_stats(mean_of([region_stats["Left"], region_stats["Right"]]), frames=False))
    if all(region_stats.values()):
        lines.append("L+R+C  : " + format_stats(mean_of(list(region_stats.values())), frames=False))
    lines.append("")
    lines.append("Per-scene Full")
    for scene, full_stats, _regions in per_scene:
        lines.append(f"{scene}: {format_stats(full_stats)}")
    return lines


def write_report(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(path, "w") as handle:
        handle.write(text)
    print(text, end="")
    print(f"Wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Pairwise HM metrics for one dataset")
    parser.add_argument("--dataset", required=True, choices=[*SPECS, "lyft"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.dataset == "lyft":
        parts = []
        for name in ("lyft1920", "lyft1224"):
            print(f"scoring {name}", flush=True)
            parts.append((name, score_spec(SPECS[name])))
        full_rows = [row for _name, (rows, _regions, _scenes, _skip) in parts for row in rows]
        region_rows = {name: [] for name in REGION_ORDER}
        per_scene = []
        skipped = 0
        extra = ["Pool: every lyft1920 and lyft1224 frame. Subsets are not averaged."]
        for name, (rows, regions, scenes, skip) in parts:
            skipped += skip
            for region in REGION_ORDER:
                region_rows[region].extend(regions[region])
            stats = mean_rows(rows)
            extra.append(f"{name}: {format_stats(stats) if stats else 'no frames'}")
            per_scene.extend(
                (f"{name}/{scene}", full_stats, region_stats)
                for scene, full_stats, region_stats in scenes
            )
        path = args.output or os.path.join(ROOT, "outputs/lyft_pairwise_hm.txt")
        write_report(path, report_lines("lyft", full_rows, region_rows, per_scene, skipped, extra))
        return

    spec = SPECS[args.dataset]
    print(f"scoring {args.dataset}", flush=True)
    full_rows, region_rows, per_scene, skipped = score_spec(spec)
    if not full_rows:
        raise SystemExit(f"no scored frames under {spec['pred']}")
    extra = [f"Pred: {spec['pred']}", f"GT: {spec['gt']}", f"GT camera: {spec['cam']}"]
    path = args.output or os.path.join(ROOT, "outputs", f"{args.dataset}_pairwise_hm.txt")
    write_report(path, report_lines(args.dataset, full_rows, region_rows, per_scene, skipped, extra))


if __name__ == "__main__":
    main()
