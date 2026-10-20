#!/usr/bin/env python3
"""Run UniStitch driving finetuning as stage 1, then stage 2.

Stage 1 trains every weight with the robustifier off. Stage 2 loads that
checkpoint, freezes the alignment backbone, and trains with the robustifier.
Each stage uses the driving epoch count: nuscenes 1, lyft 3, widedrive3 3.

    python train.py --dataset widedrive3 --gpu 5 --epochs 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASED = ROOT / "pretrained" / "epoch_best_model.pth"


def run_stage(stage, args, weight):
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_drive.py"),
        "--dataset",
        args.dataset,
        "--stage",
        stage,
        "--gpu",
        args.gpu,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.stage1_batch if stage == "stage1" else args.stage2_batch),
        "--workers",
        str(args.workers),
        "--weight",
        weight,
        "--save-dir",
        args.save_dir,
    ]
    if args.max_steps:
        command.extend(["--max-steps", str(args.max_steps)])
    print(f"\n########## {stage} ##########", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="UniStitch stage 1 then stage 2")
    parser.add_argument("--dataset", required=True, choices=("nuscenes", "lyft", "lyft1920", "lyft1224", "widedrive3"))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--stage", choices=("both", "stage1", "stage2"), default="both")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--stage1-batch", type=int, default=4, help="stage 1 trains every weight; 8 OOMs on a 24GB 4090")
    parser.add_argument("--stage2-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weight", default=str(RELEASED))
    parser.add_argument("--save-dir", default=str(ROOT / "outputs" / "finetune"))
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    names = ("lyft1920", "lyft1224") if args.dataset == "lyft" else (args.dataset,)
    for name in names:
        args.dataset = name
        stage1_last = Path(args.save_dir) / name / "stage1" / "latest.pth"
        if args.stage in ("both", "stage1"):
            run_stage("stage1", args, args.weight)
        if args.stage in ("both", "stage2"):
            weight = str(stage1_last if stage1_last.is_file() else args.weight)
            if not stage1_last.is_file():
                print(f"no stage-1 checkpoint, stage 2 starts from {weight}", flush=True)
            run_stage("stage2", args, weight)


if __name__ == "__main__":
    main()
