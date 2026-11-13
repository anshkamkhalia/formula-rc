# Adapted from TwinLiteNetPlus (train.py, utils.py):
# https://github.com/chequanghuy/TwinLiteNetPlus
#
# Copyright (c) 2025 Chế Quang Huy
# Licensed under the MIT License. See licenses/TwinLiteNet-LICENSE for full text.
#
# Modifications: npy dataset, fine-tuning from pretrained weights, device-agnostic
# (cuda/mps/cpu), autocast around the forward pass, per-track validation with
# whole-set confusion matrices, best-checkpoint selection, YAML config.

import argparse
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.experiments.lane_segmentation_ppo.lane_segmentation_fine_tune.model import TwinLiteNetPlus
from src.experiments.lane_segmentation_ppo.lane_segmentation_fine_tune.loss import TotalLoss
from src.experiments.lane_segmentation_ppo.lane_segmentation_fine_tune.IOUEval import SegmentationMetric
from src.experiments.lane_segmentation_ppo.lane_segmentation_fine_tune.dataset import (
    PAD, LaneSegDataset, build_index, make_sampler, seed_worker)

DEFAULT_CONFIG = Path(__file__).with_name("finetune.yaml")


class ModelEMA:
    """Exponential Moving Average (EMA) for model parameters"""
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config, weights=None, device="cpu"):
    model = TwinLiteNetPlus(SimpleNamespace(config=config))
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            raise RuntimeError(f"Could not load {weights} into a '{config}' model. "
                               "Check that model.config matches the checkpoint size.") from e
    return model.to(device)


def poly_lr(optimizer, base_lr, epoch, max_epochs, power):
    lr = base_lr * (1 - epoch / max_epochs) ** power
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, device, use_amp, desc):
    model.train()
    totals = {"focal": 0.0, "tversky": 0.0, "loss": 0.0}
    n = 0
    pbar = tqdm(loader, desc=desc, bar_format="{l_bar}{bar:10}{r_bar}")
    for img, (seg_da, seg_ll), _ in pbar:
        img = img.to(device, non_blocking=True).float() / 255.0
        seg_da = seg_da.to(device, non_blocking=True)
        seg_ll = seg_ll.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(img)
            focal, tversky, loss = criterion(output, (seg_da, seg_ll))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        totals["focal"] += focal
        totals["tversky"] += tversky
        totals["loss"] += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{totals['loss'] / n:.4f}")
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Per-track metrics from confusion matrices accumulated over the whole set."""
    model.eval()
    metrics, counts = {}, {}
    for img, (seg_da, seg_ll), tracks in tqdm(loader, desc="val", leave=False):
        img = img.to(device).float() / 255.0
        out_da, out_ll = model(img)
        da_pred = out_da[:, :, PAD:-PAD].argmax(1).cpu().numpy()
        ll_pred = out_ll[:, :, PAD:-PAD].argmax(1).cpu().numpy()
        da_gt = seg_da.argmax(1).numpy()
        ll_gt = seg_ll.argmax(1).numpy()
        for i, track in enumerate(tracks):
            for name in (track, "all"):
                if name not in metrics:
                    metrics[name] = (SegmentationMetric(2), SegmentationMetric(2))
                    counts[name] = 0
                metrics[name][0].addBatch(da_pred[i], da_gt[i])
                metrics[name][1].addBatch(ll_pred[i], ll_gt[i])
                counts[name] += 1

    results = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, (da_m, ll_m) in metrics.items():
            results[name] = {
                "n": counts[name],
                "da_miou": float(da_m.meanIntersectionOverUnion()),
                "ll_iou": float(ll_m.IntersectionOverUnion()),
                "ll_acc": float(ll_m.lineAccuracy()),
            }
    return results


def print_results(results):
    print(f"{'track':<30}{'frames':>8}{'DA mIoU':>10}{'LL IoU':>10}{'LL acc':>10}")
    for name in sorted(results, key=lambda t: (t == "all", t)):
        r = results[name]
        print(f"{name:<30}{r['n']:>8}{r['da_miou']:>10.3f}{r['ll_iou']:>10.3f}{r['ll_acc']:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TwinLiteNet+ on lane_seg_data")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", default=None, help="path to a run's last.pt to continue training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m, d, t = cfg["model"], cfg["data"], cfg["train"]

    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    device = get_device(t["device"])
    use_amp = t["amp"] and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    chunks, train_items, val_items = build_index(
        d["root"], d["tracks"], d["val_fraction"], d["val_only_tracks"], d["exclude_tracks"])
    if not train_items:
        raise SystemExit(f"No training frames found under {d['root']}/<track>/x*.npy")

    train_ds = LaneSegDataset(chunks, train_items, hyp=cfg["augment"], train=True)
    val_ds = LaneSegDataset(chunks, val_items) if val_items else None
    sampler = make_sampler(train_ds, d["track_weights"], d["balance_tracks"], d["samples_per_epoch"])
    if len(sampler) < t["batch_size"]:
        raise SystemExit(f"Only {len(sampler)} samples per epoch but batch_size is {t['batch_size']}")

    loader_kw = dict(num_workers=t["num_workers"], pin_memory=device.type == "cuda",
                     persistent_workers=t["num_workers"] > 0, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], sampler=sampler, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, **loader_kw) if val_ds else None

    print(f"device: {device}  amp: {use_amp}")
    val_counts = val_ds.track_counts() if val_ds else {}
    for track in sorted(set(train_ds.track_counts()) | set(val_counts)):
        print(f"  {track:<30} train {train_ds.track_counts()[track]:>6}  val {val_counts.get(track, 0):>6}")

    model = build_model(m["config"], None if args.resume else m["pretrained"], device)
    print(f"model: {m['config']}, {sum(p.numel() for p in model.parameters()):,} params")

    criterion = TotalLoss(cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=(t["momentum"], 0.999),
                                  eps=t["eps"], weight_decay=t["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_score = 0, -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_score = ckpt["epoch"], ckpt["best_score"]
        save_dir = Path(args.resume).parent

    ema = ModelEMA(model) if t["ema"] else None  # built after loading so it starts from the right weights
    if args.resume and ema is not None and ckpt["ema"] is not None:
        ema.ema.load_state_dict(ckpt["ema"])
        ema.updates = ckpt["ema_updates"]

    if not args.resume:
        save_dir = Path(t["save_dir"]) / time.strftime("%Y%m%d-%H%M%S")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"saving to {save_dir}")

    for epoch in range(start_epoch, t["epochs"]):
        lr = poly_lr(optimizer, t["lr"], epoch, t["epochs"], t["lr_power"])
        stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, ema, device, use_amp,
                                desc=f"epoch {epoch + 1}/{t['epochs']}")
        print(f"epoch {epoch + 1}: lr {lr:.2e}  loss {stats['loss']:.4f}  "
              f"(focal {stats['focal']:.4f}, tversky {stats['tversky']:.4f})")

        eval_model = ema.ema if ema is not None else model
        torch.save(eval_model.state_dict(), save_dir / "last.pth")

        is_last = epoch + 1 == t["epochs"]
        if val_loader is not None and ((epoch + 1) % t["val_every"] == 0 or is_last):
            results = evaluate(eval_model, val_loader, device)
            print_results(results)
            sel = results.get(t["select_track"], results["all"])
            score = sel[t["select_metric"]]
            if score > best_score:
                best_score = score
                torch.save(eval_model.state_dict(), save_dir / "best.pth")
                print(f"new best {t['select_metric']} on '{t['select_track']}': {score:.4f}")

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "ema": ema.ema.state_dict() if ema is not None else None,
            "ema_updates": ema.updates if ema is not None else 0,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
        }, save_dir / "last.pt")

    print(f"done. weights: {save_dir / 'best.pth'} (best), {save_dir / 'last.pth'} (last)")


if __name__ == "__main__":
    main()