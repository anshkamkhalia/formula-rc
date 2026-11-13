import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.experiments.lane_segmentation_ppo.lane_segmentation import augment as aug

IMG_H, IMG_W = 360, 640
PAD = 12  # 360 -> 384 so the encoder's 1/8 feature map divides evenly into CAAM's (2, 4) bins


def find_chunks(track_dir):
    """x.npy/y.npy, or x_000.npy/y_000.npy, x_001.npy/y_001.npy, ..."""
    pairs = []
    for x_path in sorted(track_dir.glob("x*.npy")):
        y_path = x_path.with_name("y" + x_path.name[1:])
        if not y_path.exists():
            raise FileNotFoundError(f"{x_path} has no matching {y_path.name}")
        pairs.append((x_path, y_path))
    return pairs


def check_shapes(x, y, path):
    if x.ndim != 4 or x.shape[1:] != (IMG_H, IMG_W, 3) or x.dtype != np.uint8:
        raise ValueError(f"{path}: x must be uint8 (N, {IMG_H}, {IMG_W}, 3), got {x.dtype} {x.shape}")
    if y.shape != (len(x), 2, IMG_H, IMG_W):
        raise ValueError(f"{path}: y must be (N, 2, {IMG_H}, {IMG_W}) [drivable, lane], got {y.shape}")


def build_index(root, tracks=None, val_fraction=0.15, val_only_tracks=(), exclude_tracks=()):
    """Returns (chunks, train_items, val_items). Items are (chunk_id, row)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root {root} does not exist")
    tracks = tracks or sorted(p.name for p in root.iterdir() if p.is_dir())
    tracks = [t for t in tracks if t not in exclude_tracks]

    chunks, train_items, val_items = [], [], []
    for track in tracks:
        pairs = find_chunks(root / track)
        if not pairs:
            print(f"warning: no x*.npy files in {root / track}, skipping")
            continue
        for x_path, y_path in pairs:
            x = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
            check_shapes(x, y, x_path)
            n, cid = len(x), len(chunks)
            chunks.append({"track": track, "x": str(x_path), "y": str(y_path), "n": n})

            if track in val_only_tracks:
                val_items += [(cid, i) for i in range(n)]
                continue
            # contiguous tail as val: neighbouring frames are near-duplicates, so a random split would leak
            split = n - int(round(n * val_fraction))
            train_items += [(cid, i) for i in range(split)]
            val_items += [(cid, i) for i in range(split, n)]

    return chunks, train_items, val_items


class LaneSegDataset(Dataset):
    """Returns (image uint8 [3, 384, 640], (seg_da, seg_ll) float [2, 360, 640] one-hot bg/fg, track name)."""

    def __init__(self, chunks, items, hyp=None, train=False):
        self.chunks = chunks
        self.items = items
        self.hyp = hyp
        self.train = train
        self.item_tracks = [chunks[cid]["track"] for cid, _ in items]
        self._arrays = {}

    def __getstate__(self):
        # don't pickle open memmaps into DataLoader workers; each worker reopens its own
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def _get(self, cid):
        if cid not in self._arrays:
            c = self.chunks[cid]
            self._arrays[cid] = (np.load(c["x"], mmap_mode="r"), np.load(c["y"], mmap_mode="r"))
        return self._arrays[cid]

    def track_counts(self):
        return Counter(self.item_tracks)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cid, row = self.items[idx]
        x, y = self._get(cid)
        img = np.array(x[row])
        da = (np.asarray(y[row, 0]) > 0).astype(np.uint8) * 255
        ll = (np.asarray(y[row, 1]) > 0).astype(np.uint8) * 255

        if self.train:
            img, da, ll = self._augment(img, da, ll)

        img = aug.letterbox(img, (IMG_H + 2 * PAD, IMG_W))
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))  # /255 happens on device
        return img, (self._onehot(da), self._onehot(ll)), self.chunks[cid]["track"]

    @staticmethod
    def _onehot(mask):
        fg = torch.from_numpy(mask > 127)
        return torch.stack((~fg, fg)).float()

    def _augment(self, img, da, ll):
        h = self.hyp
        if random.random() < h["prob_perspective"]:
            img, da, ll = aug.random_perspective(img, da, ll, h["degrees"], h["translate"], h["scale"], h["shear"])
        if random.random() < h["prob_hsv"]:
            img = aug.augment_hsv(img, h["hgain"], h["sgain"], h["vgain"])
        if random.random() < h["prob_flip"]:
            img, da, ll = img[:, ::-1].copy(), da[:, ::-1].copy(), ll[:, ::-1].copy()
        if random.random() < h["prob_bilateral"]:
            img = aug.random_bilateral_blur(img)
        if random.random() < h["prob_gaussian"]:
            img = aug.random_gaussian_blur(img)
        if random.random() < h["prob_jpeg"]:
            img = aug.random_jpeg(img)
        if random.random() < h["prob_noise"]:
            img = aug.random_noise(img)
        if random.random() < h["prob_crop"]:
            img, da, ll = aug.random_crop(img, da, ll, h["height_crop"], h["width_crop"])
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
            da = cv2.resize(da, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
            ll = cv2.resize(ll, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        return np.ascontiguousarray(img), da, ll


def make_sampler(dataset, track_weights=None, balance_tracks=True, num_samples=None):
    """balance_tracks=True: each track's share of an epoch is proportional to its weight (default 1),
    regardless of size. False: proportional to size * weight."""
    track_weights = track_weights or {}
    counts = dataset.track_counts()
    for t in track_weights:
        if t not in counts:
            print(f"warning: track_weights has '{t}' but there is no training data for it")
    weights = [track_weights.get(t, 1.0) / (counts[t] if balance_tracks else 1) for t in dataset.item_tracks]
    return WeightedRandomSampler(weights, num_samples or len(dataset), replacement=True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_dummy_track(root, name="dummy", n=64, seed=0):
    """Synthetic road + two lane lines, only for testing the pipeline end to end."""
    rng = np.random.default_rng(seed)
    xs = np.zeros((n, IMG_H, IMG_W, 3), np.uint8)
    ys = np.zeros((n, 2, IMG_H, IMG_W), np.uint8)
    for i in range(n):
        img = np.full((IMG_H, IMG_W, 3), int(rng.integers(40, 120)), np.uint8)
        cx = IMG_W // 2 + int(rng.integers(-80, 80))
        top_w, bot_w = int(rng.integers(60, 140)), int(rng.integers(400, 600))
        road = np.array([[cx - top_w // 2, 120], [cx + top_w // 2, 120],
                         [cx + bot_w // 2, IMG_H - 1], [cx - bot_w // 2, IMG_H - 1]], np.int32)
        da = np.zeros((IMG_H, IMG_W), np.uint8)
        cv2.fillPoly(da, [road], 1)
        img[da == 1] = int(rng.integers(60, 160))
        ll = np.zeros_like(da)
        for p0, p1 in [(road[0], road[3]), (road[1], road[2])]:
            cv2.line(ll, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1, thickness=6)
        img[ll == 1] = 230
        xs[i], ys[i, 0], ys[i, 1] = img, da, ll
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "x.npy", xs)
    np.save(out / "y.npy", ys)
    print(f"wrote {n} dummy frames to {out}")


def summarize(root, val_fraction=0.15):
    chunks, train_items, val_items = build_index(root, val_fraction=val_fraction)
    train_c = Counter(chunks[c]["track"] for c, _ in train_items)
    val_c = Counter(chunks[c]["track"] for c, _ in val_items)
    print(f"{'track':<30}{'chunks':>8}{'train':>8}{'val':>8}{'drivable %':>12}{'lane %':>9}")
    for track in sorted({c["track"] for c in chunks}):
        track_chunks = [c for c in chunks if c["track"] == track]
        y = np.load(track_chunks[0]["y"], mmap_mode="r")
        sample = y[::max(1, len(y) // 20)]
        print(f"{track:<30}{len(track_chunks):>8}{train_c[track]:>8}{val_c[track]:>8}"
              f"{100 * (sample[:, 0] > 0).mean():>12.1f}{100 * (sample[:, 1] > 0).mean():>9.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect lane_seg_data or generate a dummy track")
    parser.add_argument("--root", default="lane_seg_data")
    parser.add_argument("--dummy", action="store_true", help="write a synthetic track to <root>/dummy")
    args = parser.parse_args()
    if args.dummy:
        make_dummy_track(args.root)
    summarize(args.root)