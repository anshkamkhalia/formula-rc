import cv2 as cv
import numpy as np
import os

SIM_ROOT = "vae_data"
REAL_ROOT = "vae_data/real-racetrack"
OUT = "vae_data/merged/merged.npy"
SKIP = {".DS_Store", "real-racetrack", "merged"}
IMG_EXTS = (".jpg", ".jpeg", ".png")

# collect file paths
sim_files = []
for track in os.listdir(SIM_ROOT):
    if track in SKIP:
        continue
    track_dir = os.path.join(SIM_ROOT, track)
    for f in os.listdir(track_dir):
        if f.endswith(".npy"):
            sim_files.append(os.path.join(track_dir, f))

real_files = []
for split in ["test", "valid", "train"]:
    d = os.path.join(REAL_ROOT, split, "images")
    for f in os.listdir(d):
        if f.lower().endswith(IMG_EXTS):
            real_files.append(os.path.join(d, f))

total = 0
frame_shape, dtype = None, None
for p in sim_files:
    a = np.load(p, mmap_mode="r")
    if frame_shape is None:
        frame_shape, dtype = a.shape[1:], a.dtype
    elif a.shape[1:] != frame_shape or a.dtype != dtype:
        raise ValueError(f"{p}: {a.shape[1:]} {a.dtype} != {frame_shape} {dtype}")
    total += len(a)
    del a
total += len(real_files)

print(f"{total} frames of shape {frame_shape}, dtype {dtype}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
out = np.lib.format.open_memmap(OUT, mode="w+", dtype=dtype, shape=(total, *frame_shape))

i = 0
for p in sim_files:
    a = np.load(p, mmap_mode="r")
    out[i:i + len(a)] = a
    i += len(a)
    del a

for p in real_files:
    img = cv.imread(p)
    if img is None:
        raise ValueError(f"could not read {p}")
    out[i] = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    i += 1

out.flush()
del out
print("saved", OUT)