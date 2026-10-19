#!/usr/bin/env python3
"""Stitch three rear cameras with UniStitch on a white background.

chain:    cam5 with cam4, then that panorama with cam3
pairwise: cam5+cam4 and cam5+cam3, then shift the two panoramas on cam5

Saved per frame, in that order:
  stitch/  white panorama, for visualization
  crop/    cam5 vertical span, full stitched width, resized to sparse GT, for metrics

Metric crop, not a registration:
  vertical span is camera 5's axis-aligned bounding box;
  left and right extend to the maximum stitched image;
  that crop is resized to the sparse wide-FOV GT size.

The released checkpoint's point encoder takes a 256-d descriptor (stem is
2+256 channels), the same width as SuperPoint. Matches come from SIFT; the
128-d descriptor is repeated to 256. Geometry is the matched point locations.

Examples:
  python stitch_drive.py --dataset nuscenes --scene 078 --metric
  python stitch_drive.py --dataset nuscenes --scene 078 --frame 19
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Codes"))

from network import Network, build_output_model  # noqa: E402

WORK = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub")
DATASETS = {
    "nuscenes": WORK / "datasets/nuscenes/processed_10Hz_v2",
    "lyft1920": WORK / "datasets/lyft/lyft_val1920_3cams",
    "lyft1224": WORK / "datasets/lyft/lyft_val1224_3cams",
    "widedrive3": WORK / "datasets/WideDrive3_processed/WideDriveVal",
    "ddad": WORK / "datasets/ddad_process/train",
    "syn_nuscenes": WORK / "datasets/syn_nuscenes/processed_10Hz",
}
CKPT = ROOT / "pretrained/epoch_best_model.pth"
DESC_DIM = 256
MAX_POINTS = 2048
WHITE = (255, 255, 255)
GT_SPECS = {
    "nuscenes": {
        "root": WORK / "datasets/nuscenes/sparseWideFOVImages3_1554x294",
        "size": (1554, 294),
        "cam": 5,
        "suffix": "_5_sparse_wide.png",
        "val_list": WORK / "datasets/nuscenes/processed_10Hz_v2/nuScenes_Val.txt",
    },
    "lyft1920": {
        "root": WORK / "datasets/lyft/1920_sparseWideFOVImages3_1554x294",
        "size": (1554, 294),
        "cam": 5,
        "suffix": "_5_sparse_wide.png",
        "val_list": WORK / "datasets/lyft/lyft_val1920_3cams/lyft_val1920.txt",
    },
    "lyft1224": {
        "root": WORK / "datasets/lyft/1224_sparseWideFOVImages3_1554x434",
        "size": (1554, 434),
        "cam": 5,
        "suffix": "_5_sparse_wide.png",
        "val_list": WORK / "datasets/lyft/lyft_val1224_3cams/lyft_val1224.txt",
    },
    "widedrive3": {
        "root": WORK / "datasets/WideDrive_processed/sparseWideFOVImages3_1554x294",
        "size": (1554, 294),
        "cam": 2,
        "suffix": "_2_sparse_wide.png",
        "val_list": WORK / "datasets/WideDrive3_processed/WideDriveVal/val.txt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="UniStitch 3-camera driving stitch")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--scene", default=None, help="scene id, e.g. 78 or 078 or 004")
    parser.add_argument("--all", action="store_true", help="every scene in the dataset val list")
    parser.add_argument("--frame", action="append", default=None, help="frame id, repeatable")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--max-height", type=int, default=480)
    parser.add_argument("--max-out-height", type=int, default=6000)
    parser.add_argument("--save-root", default=str(ROOT / "outputs"))
    parser.add_argument("--metric", action="store_true", help="save white stitches resized to sparse GT")
    parser.add_argument("--methods", nargs="+", default=["chain", "pairwise"], choices=["chain", "pairwise"])
    parser.add_argument("--metric-dir", default=None, help="override the crop folder used for metrics")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--gt-width", type=int, default=None)
    parser.add_argument("--gt-height", type=int, default=None)
    return parser.parse_args()


def scene_dir(dataset, scene):
    root = DATASETS[dataset]
    name = str(scene)
    candidates = [name, name.zfill(3)]
    if name.isdigit():
        candidates.append(str(int(name)).zfill(3))
    for cand in candidates:
        path = root / cand
        if path.is_dir():
            return path
    raise SystemExit(f"scene not found under {root}: tried {candidates}")


def frame_token(frame):
    text = str(frame)
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def load_cam(scene, frame, cam):
    token = frame_token(frame)
    path = scene / "images" / f"{token}_{cam}.jpg"
    if not path.is_file():
        alt = scene / "images" / f"{token}_{cam}.png"
        path = alt if alt.is_file() else path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"missing image: {path}")
    return image, path


def resize_height(image, height):
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, int(round(w * (height / h))))
    interp = cv2.INTER_AREA if height < h else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)


def sift_pair(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    empty = (
        np.zeros((0, 2), np.float32),
        np.zeros((0, 2), np.float32),
        np.zeros((0, DESC_DIM), np.float32),
        np.zeros((0, DESC_DIM), np.float32),
    )
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return empty
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    d1 = des1[[m.queryIdx for m in good]].astype(np.float32)
    d2 = des2[[m.trainIdx for m in good]].astype(np.float32)
    # Checkpoint descriptor width is 256. SIFT is 128, so repeat it.
    d1 = np.concatenate([d1, d1], axis=1)
    d2 = np.concatenate([d2, d2], axis=1)
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True) + 1e-6
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True) + 1e-6
    return pts1, pts2, d1, d2


def pad_points(points, desc):
    if len(points) == 0:
        return torch.zeros(MAX_POINTS, 2), torch.zeros(MAX_POINTS, DESC_DIM)
    points = torch.from_numpy(points)
    desc = torch.from_numpy(desc)
    n = points.shape[0]
    if n >= MAX_POINTS:
        return points[:MAX_POINTS], desc[:MAX_POINTS]
    extra = torch.randint(0, n, (MAX_POINTS - n,))
    return torch.cat([points, points[extra]], 0), torch.cat([desc, desc[extra]], 0)


def to_tensor(image, points, desc):
    h, w = image.shape[:2]
    norm = image.astype(np.float32)
    norm = (norm / 127.5) - 1.0
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
    pts, des = pad_points(points, desc)
    pts = pts.clone()
    pts[:, 0] /= max(w - 1, 1)
    pts[:, 1] /= max(h - 1, 1)
    return tensor, pts.unsqueeze(0), des.unsqueeze(0)


def blend_masked(ref, tgt, mask_ref, mask_tgt):
    """Keep a short seam. Weight each side by distance to its own mask edge."""
    mask_ref = mask_ref.astype(np.uint8)
    mask_tgt = mask_tgt.astype(np.uint8)
    weight_ref = cv2.distanceTransform(mask_ref, cv2.DIST_L2, 5).astype(np.float32) ** 2
    weight_tgt = cv2.distanceTransform(mask_tgt, cv2.DIST_L2, 5).astype(np.float32) ** 2
    weight = weight_ref + weight_tgt + 1e-6
    fused = ref * (weight_ref / weight)[..., None] + tgt * (weight_tgt / weight)[..., None]
    valid = (mask_ref | mask_tgt).astype(bool)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    fused[~valid] = 255
    return fused, valid


def fuse(out):
    ref = out["output_tps_ref"][0, 0:3].detach().float().cpu().numpy().transpose(1, 2, 0)
    tgt = out["output_tps_tgt"][0, 0:3].detach().float().cpu().numpy().transpose(1, 2, 0)
    # Network output is (input+1), so *127.5 brings it back to 0..255.
    ref = np.clip(ref * 127.5, 0, 255)
    tgt = np.clip(tgt * 127.5, 0, 255)
    mask_ref = out["output_tps_ref"][0, 3].detach().float().cpu().numpy() > 0.5
    mask_tgt = out["output_tps_tgt"][0, 3].detach().float().cpu().numpy() > 0.5
    fused, valid = blend_masked(ref, tgt, mask_ref, mask_tgt)
    cam5 = out["output_cam5"][0, 0].detach().float().cpu().numpy() > 0.5
    return fused, valid, cam5


def crop_valid(image, mask, margin=2):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image, mask
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def crop_valid_bundle(image, valid, cam5, margin=2):
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return image, valid, cam5
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    return image[y0:y1, x0:x1], valid[y0:y1, x0:x1], cam5[y0:y1, x0:x1]


def crop_cam5_lr_max(image, valid, cam5):
    """Vertical: cam5 bounding box. Horizontal: maximum stitched image."""
    ys, xs = np.where(cam5)
    if len(xs) == 0:
        raise RuntimeError("camera 5 mask is empty")
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    content_ys, content_xs = np.where(valid)
    if len(content_xs) == 0:
        raise RuntimeError("stitched image is empty")
    x0 = int(content_xs.min())
    x1 = int(content_xs.max()) + 1
    y0 = max(0, min(y0, image.shape[0] - 1))
    y1 = max(y0 + 1, min(image.shape[0], y1))
    x0 = max(0, min(x0, image.shape[1] - 1))
    x1 = max(x0 + 1, min(image.shape[1], x1))
    return image[y0:y1, x0:x1]


def resize_to(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    shrinking = image.shape[1] > width or image.shape[0] > height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_mask(mask, width, height):
    if mask.shape[1] == width and mask.shape[0] == height:
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def translation(base, other, mask_base, mask_other):
    """Shift that maps an other-pixel onto the base image. Returns dx, dy, inliers."""
    sift = cv2.SIFT_create(nfeatures=3000)
    gray_base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray_other = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray_base, mask_base.astype(np.uint8) * 255)
    kp2, des2 = sift.detectAndCompute(gray_other, mask_other.astype(np.uint8) * 255)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    pts_base = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_other = np.float32([kp2[m.trainIdx].pt for m in good])
    delta = pts_base - pts_other
    median = np.median(delta, axis=0)
    error = np.linalg.norm(delta - median, axis=1)
    inliers = error < 15
    if int(inliers.sum()) < 10:
        return None
    shift = np.median(delta[inliers], axis=0)
    return float(shift[0]), float(shift[1]), int(inliers.sum())


def place_layer(canvas_h, canvas_w, image, mask, origin):
    color = np.zeros((canvas_h, canvas_w, 3), np.float32)
    weight = np.zeros((canvas_h, canvas_w), np.float32)
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    height, width = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas_w, x + width), min(canvas_h, y + height)
    if x1 <= x0 or y1 <= y0:
        return color, weight
    ix0, iy0 = x0 - x, y0 - y
    ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)
    sub_mask = mask[iy0:iy1, ix0:ix1].astype(np.uint8)
    if sub_mask.max() == 0:
        return color, weight
    dist = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5).astype(np.float32) ** 2
    color[y0:y1, x0:x1] = image[iy0:iy1, ix0:ix1].astype(np.float32) * dist[..., None]
    weight[y0:y1, x0:x1] = dist
    return color, weight


def paste_mask(canvas_h, canvas_w, mask, origin):
    out = np.zeros((canvas_h, canvas_w), np.uint8)
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    height, width = mask.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas_w, x + width), min(canvas_h, y + height)
    if x1 <= x0 or y1 <= y0:
        return out > 0
    ix0, iy0 = x0 - x, y0 - y
    ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)
    out[y0:y1, x0:x1] = mask[iy0:iy1, ix0:ix1].astype(np.uint8)
    return out > 0


def compose_panoramas(base, mask_base, cam5_base, other, mask_other, cam5_other):
    """Shift the cam5+cam3 panorama onto the cam5+cam4 panorama."""
    found = translation(base, other, mask_base, mask_other)
    if found is None:
        print("  compose alignment failed, placing cam3 panorama on the left")
        dx, dy, inliers = float(-other.shape[1] // 2), float((base.shape[0] - other.shape[0]) / 2), 0
    else:
        dx, dy, inliers = found
        print(f"  compose shift=({dx:.1f},{dy:.1f}) inliers={inliers}")
    width_base, height_base = base.shape[1], base.shape[0]
    width_other, height_other = other.shape[1], other.shape[0]
    x0 = min(0.0, dx)
    y0 = min(0.0, dy)
    x1 = max(float(width_base), dx + width_other)
    y1 = max(float(height_base), dy + height_other)
    canvas_w = int(round(x1 - x0))
    canvas_h = int(round(y1 - y0))
    origin_base = (-x0, -y0)
    origin_other = (dx - x0, dy - y0)
    color_base, weight_base = place_layer(canvas_h, canvas_w, base, mask_base, origin_base)
    color_other, weight_other = place_layer(canvas_h, canvas_w, other, mask_other, origin_other)
    weight = weight_base + weight_other
    valid = weight > 0
    fused = np.full((canvas_h, canvas_w, 3), 255, np.float32)
    fused[valid] = (color_base[valid] + color_other[valid]) / weight[valid, None]
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    cam5 = paste_mask(canvas_h, canvas_w, cam5_base, origin_base) | paste_mask(
        canvas_h, canvas_w, cam5_other, origin_other
    )
    fused, valid, cam5 = crop_valid_bundle(fused, valid, cam5)
    return fused, valid, cam5, inliers > 0


def load_net(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not torch.cuda.is_available():
        raise SystemExit("UniStitch warping calls .cuda(); no GPU is visible")
    net = Network(descriptor_dim=DESC_DIM)
    ckpt = torch.load(CKPT, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state, strict=True)
    net = net.cuda().eval()
    net.fuse()
    return net


@torch.no_grad()
def stitch_pair(net, image1, image2, max_height, cam5_mask=None, max_out_height=6000, strict=False):
    if image1.shape[:2] != image2.shape[:2]:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
    image1 = resize_height(image1, min(max_height, image1.shape[0]))
    image2 = resize_height(image2, image1.shape[0])
    if image2.shape[1] != image1.shape[1]:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
    if cam5_mask is not None:
        cam5_mask = resize_mask(cam5_mask, image1.shape[1], image1.shape[0])
    pts1, pts2, des1, des2 = sift_pair(image1, image2)
    print(f"  matches={len(pts1)} size={image1.shape[1]}x{image1.shape[0]}")
    t1, p1, d1 = to_tensor(image1, pts1, des1)
    t2, p2, d2 = to_tensor(image2, pts2, des2)
    cam5_t = None
    if cam5_mask is not None:
        cam5_t = torch.from_numpy(np.ascontiguousarray(cam5_mask.astype(np.float32))).view(
            1, 1, cam5_mask.shape[0], cam5_mask.shape[1]
        )
    batch, ok = build_output_model(
        net, t1.cuda(), t2.cuda(), p1.cuda(), p2.cuda(), d1.cuda(), d2.cuda(),
        max_out_height=max_out_height, cam5_tensor=None if cam5_t is None else cam5_t.cuda(),
    )
    if not ok or batch is None:
        if strict:
            raise RuntimeError(f"warp canvas is larger than {max_out_height}")
        print("  warp canvas too large, using side-by-side fallback")
        side = np.concatenate([image2, image1], axis=1)
        valid = np.ones(side.shape[:2], dtype=bool)
        cam5 = np.zeros(side.shape[:2], dtype=bool)
        cam5[:, image2.shape[1]:] = True
        return side, valid, cam5, False
    fused, valid, cam5 = fuse(batch)
    fused, valid, cam5 = crop_valid_bundle(fused, valid, cam5)
    return fused, valid, cam5, True


def stitch_chain(net, images, max_height, max_out_height):
    fused, valid, cam5, _ = stitch_pair(
        net, images[5], images[4], max_height, max_out_height=max_out_height, strict=True
    )
    fused, valid, cam5 = crop_valid_bundle(fused, valid, cam5)
    fused, valid, cam5, _ = stitch_pair(
        net, fused, images[3], max_height, cam5_mask=cam5, max_out_height=max_out_height, strict=True
    )
    return crop_valid_bundle(fused, valid, cam5)


def stitch_pairwise(net, images, max_height, max_out_height):
    left, left_valid, left_cam5, _ = stitch_pair(
        net, images[5], images[4], max_height, max_out_height=max_out_height, strict=True
    )
    left, left_valid, left_cam5 = crop_valid_bundle(left, left_valid, left_cam5)
    right, right_valid, right_cam5, _ = stitch_pair(
        net, images[5], images[3], max_height, max_out_height=max_out_height, strict=True
    )
    right, right_valid, right_cam5 = crop_valid_bundle(right, right_valid, right_cam5)
    fused, valid, cam5, _ = compose_panoramas(
        left, left_valid, left_cam5, right, right_valid, right_cam5
    )
    return fused, valid, cam5


def montage(cam3, cam4, cam5, step1, final):
    target_h = 240

    def row(images):
        resized = [resize_height(im, target_h) for im in images]
        return np.concatenate(resized, axis=1)

    top = row([cam3, cam5, cam4])
    bottom = row([step1, final])
    width = max(top.shape[1], bottom.shape[1])

    def pad(image):
        if image.shape[1] == width:
            return image
        canvas = np.zeros((image.shape[0], width, 3), dtype=image.dtype)
        canvas[:, : image.shape[1]] = image
        return canvas

    return np.concatenate([pad(top), pad(bottom)], axis=0)


def stitch_frame(net, dataset, scene, frame, save_root, max_height):
    cam5, p5 = load_cam(scene, frame, 5)
    cam4, p4 = load_cam(scene, frame, 4)
    cam3, p3 = load_cam(scene, frame, 3)
    print(f"{dataset} {scene.name} frame {frame_token(frame)}")
    print(f"  inputs {p5.name} {p4.name} {p3.name} {cam5.shape[1]}x{cam5.shape[0]}")
    height = min(max_height, cam5.shape[0])
    cam5_s = resize_height(cam5, height)
    cam4_s = resize_height(cam4, height)
    cam3_s = resize_height(cam3, height)
    print("  stitch cam5 + cam4")
    step1, mask1, cam5_1, ok1 = stitch_pair(net, cam5_s, cam4_s, height)
    print("  stitch cam5 + cam3")
    step3, mask3, cam5_3, ok3 = stitch_pair(net, cam5_s, cam3_s, height)
    print("  align the two panoramas on cam5")
    final, valid, cam5, ok2 = compose_panoramas(step1, mask1, cam5_1, step3, mask3, cam5_3)
    out_dir = Path(save_root) / dataset / scene.name / f"frame_{frame_token(frame)}"
    if dataset in GT_SPECS:
        spec = gt_spec(dataset)
        stitch_shape, crop_shape = save_outputs(final, valid, cam5, spec["size"][0], spec["size"][1], out_dir, frame_token(frame))
        print(f"  saved stitch {stitch_shape[1]}x{stitch_shape[0]} crop {crop_shape[1]}x{crop_shape[0]} ok=({ok1},{ok3},{ok2})")
    else:
        stitch = final.copy()
        stitch[~valid] = 255
        cropped = crop_cam5_lr_max(stitch, valid, cam5)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "stitched.jpg"), stitch)
        cv2.imwrite(str(out_dir / "crop.png"), cropped)
        print(f"  saved {out_dir / 'stitched.jpg'} and {out_dir / 'crop.png'} ok=({ok1},{ok3},{ok2})")
    return out_dir


def gt_spec(dataset):
    if dataset not in GT_SPECS:
        raise SystemExit(f"no sparse wide-FOV GT for {dataset}")
    spec = dict(GT_SPECS[dataset])
    local = ROOT / "datasets" / Path(spec["root"]).relative_to(WORK / "datasets")
    if local.is_dir():
        spec["root"] = local
    return spec


def read_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def list_metric_frames(scene, gt_dir, suffix, ref_cam=5):
    image_dir = scene / "images"
    frames = []
    for name in sorted(os.listdir(image_dir)):
        if not name.endswith(f"_{ref_cam}.jpg"):
            continue
        frame = name[: -len(f"_{ref_cam}.jpg")]
        paths = [image_dir / f"{frame}_{view}.jpg" for view in (5, 4, 3)]
        gt_path = gt_dir / "rgb" / f"{frame}{suffix}"
        mask_path = gt_dir / "mask" / f"{frame}{suffix}"
        if all(path.is_file() for path in paths) and gt_path.is_file() and mask_path.is_file():
            frames.append(frame)
    return frames


def save_outputs(image, valid, cam5, gt_w, gt_h, out_dir, frame):
    """Write the white stitch first, then the GT-sized crop used for metrics."""
    if max(image.shape[:2]) > 65500:
        raise RuntimeError(f"canvas {image.shape[1]}x{image.shape[0]} exceeds the image limit")
    stitch = image.copy()
    stitch[~valid] = 255
    cropped = crop_cam5_lr_max(stitch, valid, cam5)
    resized = resize_to(cropped, gt_w, gt_h)
    stitch_dir = Path(out_dir) / "stitch"
    crop_dir = Path(out_dir) / "crop"
    stitch_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(stitch_dir / f"{frame}.jpg"), stitch)
    cv2.imwrite(str(crop_dir / f"{frame}.png"), resized)
    return stitch.shape, resized.shape


def outputs_ready(out_dir, frame):
    return (Path(out_dir) / "stitch" / f"{frame}.jpg").is_file() and (
        Path(out_dir) / "crop" / f"{frame}.png"
    ).is_file()


def run_metric(args):
    spec = gt_spec(args.dataset)
    gt_w, gt_h = spec["size"]
    if args.gt_width is not None:
        gt_w = args.gt_width
    if args.gt_height is not None:
        gt_h = args.gt_height
    if args.all:
        scene_names = read_list(spec["val_list"])
    elif args.scene:
        scene_names = [args.scene]
    else:
        raise SystemExit("pass --scene or --all")
    net = load_net(args.gpu)
    print(
        f"{args.dataset}: {len(scene_names)} scenes -> {gt_w}x{gt_h}, methods {args.methods}",
        flush=True,
    )
    totals = {method: {"saved": 0, "skipped": 0, "failed": 0} for method in args.methods}
    for scene_index, scene_name in enumerate(scene_names):
        scene = scene_dir(args.dataset, scene_name)
        gt_dir = spec["root"] / scene.name
        if args.frame:
            frames = [frame_token(frame) for frame in args.frame]
        else:
            frames = list_metric_frames(scene, gt_dir, spec["suffix"])
        if args.max_frames > 0:
            frames = frames[: args.max_frames]
        if not frames:
            print(f"scene {scene.name}: no frames", flush=True)
            continue
        print(f"scene {scene_index + 1}/{len(scene_names)} {scene.name}: {len(frames)} frames", flush=True)
        for method in args.methods:
            saved = skipped = failed = 0
            for index, frame in enumerate(frames):
                out_dir = Path(args.save_root) / args.dataset / scene.name / method
                if args.metric_dir:
                    out_dir = Path(args.metric_dir) / args.dataset / method / scene.name
                if outputs_ready(out_dir, frame):
                    skipped += 1
                    continue
                try:
                    images = {cam: load_cam(scene, frame, cam)[0] for cam in (5, 4, 3)}
                    if method == "chain":
                        image, valid, cam5 = stitch_chain(net, images, args.max_height, args.max_out_height)
                    else:
                        image, valid, cam5 = stitch_pairwise(net, images, args.max_height, args.max_out_height)
                    stitch_shape, metric_shape = save_outputs(
                        image, valid, cam5, gt_w, gt_h, out_dir, frame
                    )
                except KeyboardInterrupt:
                    raise
                except (Exception, SystemExit) as exc:
                    failed += 1
                    print(f"  skip {method} {scene.name} {frame}: {exc}", flush=True)
                    continue
                saved += 1
                if saved == 1:
                    print(
                        f"  {method} {frame}: stitch {stitch_shape[1]}x{stitch_shape[0]} "
                        f"-> crop {metric_shape[1]}x{metric_shape[0]}",
                        flush=True,
                    )
                if (index + 1) % 20 == 0 or index + 1 == len(frames):
                    print(f"  {method} {index + 1}/{len(frames)} frame {frame}", flush=True)
            for key, count in (("saved", saved), ("skipped", skipped), ("failed", failed)):
                totals[method][key] += count
            print(f"  {method}: saved {saved}, skipped {skipped}, failed {failed}", flush=True)
    for method, counts in totals.items():
        print(
            f"{method} total: saved {counts['saved']}, skipped {counts['skipped']}, failed {counts['failed']}",
            flush=True,
        )


def main():
    args = parse_args()
    if args.metric:
        run_metric(args)
        return
    if not args.scene:
        raise SystemExit("pass --scene")
    if not args.frame:
        raise SystemExit("pass --frame, or --metric to score a scene")
    scene = scene_dir(args.dataset, args.scene)
    net = load_net(args.gpu)
    for frame in args.frame:
        stitch_frame(net, args.dataset, scene, frame, args.save_root, args.max_height)


if __name__ == "__main__":
    main()
