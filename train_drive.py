#!/usr/bin/env python3
"""Finetune UniStitch on driving pairs, using the released stage-2 recipe.

Each sample is two images only: cam5+cam4 or cam5+cam3. Train scenes and
frames are shuffled every epoch. The released checkpoint is loaded, the point
backbone and both ResNet feature extractors stay frozen, and the rest is
trained with AdamW 1e-4 and ExponentialLR 0.97. The loss is the original
overlap, mesh, and descriptor-fusion loss. SIFT matches are computed on the
512 training image and repeated from 128-d to 256-d, the same descriptor
width as the checkpoint.

Defaults start from pretrained/epoch_best_model.pth:
  nuscenes: 1 epoch
  lyft: 1920 and 1224 mixed, 3 epochs
  widedrive3: 3 epochs on WideDriveTrain/train.txt, beside WideDriveVal
"""
import os
import sys

_GPU = "1"
if "--gpu" in sys.argv:
    _GPU = sys.argv[sys.argv.index("--gpu") + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU)

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Codes"))
sys.path.insert(0, str(ROOT))

from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss  # noqa: E402
from network import Network, build_model  # noqa: E402
from stitch_drive import WORK, load_cam, pad_points, sift_pair  # noqa: E402

CKPT = ROOT / "pretrained/epoch_best_model.pth"
TRAIN_SPECS = {
    "nuscenes": {
        "roots": [WORK / "datasets/nuscenes/processed_10Hz_v2"],
        "list": WORK / "datasets/nuscenes/processed_10Hz_v2/nuScenes_Train.txt",
        "epochs": 1,
    },
    "lyft1920": {
        "roots": [WORK / "datasets/lyft/lyft_train1920_3cams"],
        "list": None,
        "epochs": 3,
    },
    "lyft1224": {
        "roots": [WORK / "datasets/lyft/lyft_train1224_3cams"],
        "list": None,
        "epochs": 3,
    },
    "lyft": {
        "roots": [
            WORK / "datasets/lyft/lyft_train1920_3cams",
            WORK / "datasets/lyft/lyft_train1224_3cams",
        ],
        "list": None,
        "epochs": 3,
    },
    "widedrive3": {
        "roots": [WORK / "datasets/WideDrive3_processed/WideDriveTrain"],
        "list": WORK / "datasets/WideDrive3_processed/WideDriveTrain/train.txt",
        "epochs": 3,
    },
}
SIDE_CAMS = (4, 3)


def read_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def scene_names(spec):
    if spec["list"] is not None:
        if not spec["list"].is_file():
            raise SystemExit(f"split list not found: {spec['list']}")
        return read_list(spec["list"])
    names = []
    for root in spec["roots"]:
        names.extend(
            path.name for path in sorted(root.iterdir()) if (path / "images").is_dir()
        )
    return names


def resolve_scene(roots, name):
    candidates = [name]
    if name.isdigit():
        candidates.append(str(int(name)).zfill(3))
    for root in roots:
        for cand in candidates:
            path = root / cand
            if (path / "images").is_dir():
                return path
    return None


class DrivingPairDataset(Dataset):
    """One item is cam5 plus either cam4 or cam3, from the train scenes."""

    def __init__(self, dataset, size=512):
        spec = TRAIN_SPECS[dataset]
        self.size = size
        self.pairs = []
        missing = 0
        for name in scene_names(spec):
            scene = resolve_scene(spec["roots"], name)
            if scene is None:
                missing += 1
                continue
            image_dir = scene / "images"
            frames = []
            for filename in os.listdir(image_dir):
                if filename.endswith("_5.jpg"):
                    frames.append(filename[: -len("_5.jpg")])
            for frame in frames:
                for side in SIDE_CAMS:
                    side_path = image_dir / f"{frame}_{side}.jpg"
                    if side_path.is_file():
                        self.pairs.append((scene, frame, side))
        if not self.pairs:
            raise SystemExit(f"no cam5/cam4 or cam5/cam3 pairs for {dataset}")
        print(
            f"{dataset}: {len(self.pairs)} pairs, {missing} listed scenes missing",
            flush=True,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        scene, frame, side = self.pairs[index]
        image1, _ = load_cam(scene, frame, 5)
        image2, _ = load_cam(scene, frame, side)
        image1 = cv2.resize(image1, (self.size, self.size), interpolation=cv2.INTER_AREA)
        image2 = cv2.resize(image2, (self.size, self.size), interpolation=cv2.INTER_AREA)
        pts1, pts2, des1, des2 = sift_pair(image1, image2)
        tensor1, point1, desc1 = self._pack(image1, pts1, des1)
        tensor2, point2, desc2 = self._pack(image2, pts2, des2)
        if random.randint(0, 1) == 1:
            tensor1, tensor2 = tensor2, tensor1
            point1, point2 = point2, point1
            desc1, desc2 = desc2, desc1
        return tensor1, tensor2, point1, point2, desc1, desc2

    def _pack(self, image, points, desc):
        norm = image.astype(np.float32)
        norm = (norm / 127.5) - 1.0
        tensor = torch.from_numpy(np.ascontiguousarray(norm.transpose(2, 0, 1)))
        pts, des = pad_points(points, desc)
        pts = pts.clone()
        pts[:, 0] /= max(self.size - 1, 1)
        pts[:, 1] /= max(self.size - 1, 1)
        return tensor, pts, des


def load_net(weight, freeze):
    net = Network(descriptor_dim=256)
    checkpoint = torch.load(weight, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    net.load_state_dict(state, strict=True)
    if freeze:
        net.point_backbone.requires_grad_(False)
        net.feature_extractor_stage1.requires_grad_(False)
        net.feature_extractor_stage2.requires_grad_(False)
    net.cuda()
    trainable = sum(param.numel() for param in net.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in net.parameters() if not param.requires_grad)
    print(f"loaded {weight} trainable={trainable} frozen={frozen}", flush=True)
    return net


def train(args):
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = DrivingPairDataset(args.dataset, size=args.size)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    if len(loader) == 0:
        raise SystemExit("batch size is larger than the pair set")
    net = load_net(args.weight, freeze=not args.unfreeze)
    net.train()
    params = [param for param in net.parameters() if param.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    save_dir = Path(args.save_dir) / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    glob_iter = 0
    latest = save_dir / "latest.pth"
    if args.resume and latest.is_file():
        checkpoint = torch.load(latest, map_location="cpu")
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        glob_iter = checkpoint["glob_iter"]
        for _ in range(start_epoch):
            scheduler.step()
        print(f"resume {latest} epoch {start_epoch}", flush=True)

    steps_per_epoch = len(loader)
    print(
        f"epochs {start_epoch}->{args.epochs} steps/epoch {steps_per_epoch} "
        f"batch {args.batch_size} lr {args.lr}",
        flush=True,
    )
    stop = False
    for epoch in range(start_epoch, args.epochs):
        loss_sum = overlap_sum = mesh_sum = 0.0
        seen = 0
        epoch_start = time.time()
        print(f"epoch {epoch} lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)
        for step, batch in enumerate(loader):
            images1, images2, points1, points2, desc1, desc2 = [
                item.float().cuda(non_blocking=True) for item in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            out = build_model(
                net, images1, images2, points1, points2, desc1, desc2, is_stage2=True
            )
            overlap = cal_lp_loss(
                out["output_H_ref"], out["output_H_tgt"], out["output_tps_ref"], out["output_tps_tgt"]
            )
            mesh = 10 * (
                inter_grid_loss(out["mesh_ref"])
                + intra_grid_loss(out["mesh_ref"])
                + inter_grid_loss(out["mesh_tgt"])
                + intra_grid_loss(out["mesh_tgt"])
            )
            total = overlap + mesh + out["df_loss"]
            total.backward()
            clip_grad_norm_(params, max_norm=3, norm_type=2)
            optimizer.step()
            loss_sum += float(total.item())
            overlap_sum += float(overlap.item())
            mesh_sum += float(mesh.item())
            seen += 1
            glob_iter += 1
            if step == 0 or (step + 1) % args.log_every == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"  epoch {epoch + 1}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                    f"loss {loss_sum / seen:.4f} overlap {overlap_sum / seen:.4f} "
                    f"mesh {mesh_sum / seen:.4f} {elapsed / seen:.3f}s/step",
                    flush=True,
                )
            if args.max_steps and glob_iter >= args.max_steps:
                stop = True
                break
        scheduler.step()
        if not args.no_save:
            state = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "glob_iter": glob_iter,
                "dataset": args.dataset,
            }
            torch.save(state, latest)
            torch.save(state, save_dir / f"epoch_{epoch + 1:03d}.pth")
        print(
            f"epoch {epoch + 1} done in {time.time() - epoch_start:.1f}s "
            f"loss {loss_sum / max(seen, 1):.4f}",
            flush=True,
        )
        if stop:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune UniStitch on cam5 pairs")
    parser.add_argument("--dataset", required=True, choices=sorted(TRAIN_SPECS))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--batch-size", type=int, default=8, help="stable batch on a 24GB 4090; 11 OOMs on the next step")
    parser.add_argument("--epochs", type=int, default=None, help="default: nuscenes 1, lyft 3, widedrive3 3")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-gamma", type=float, default=0.97)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weight", default=str(CKPT))
    parser.add_argument("--save-dir", default=str(ROOT / "outputs" / "finetune"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--unfreeze", action="store_true", help="train every weight, not the stage-2 freeze")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many updates, for a timing run")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = TRAIN_SPECS[args.dataset]["epochs"]
    return args


if __name__ == "__main__":
    train(parse_args())
    # Torch 1.13 workers abort in their C++ teardown after a normal exit.
    os._exit(0)
