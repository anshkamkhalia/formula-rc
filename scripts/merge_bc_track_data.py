import os
import numpy as np
import gc

data_path = "bc_data"
files = os.listdir(data_path)

# keys are frames, speeds, actions

total_samples = 0

merged_frames = []
merged_speeds = []
merged_actions = []

for file in files:
    if file == ".DS_Store":
        continue
    with np.load(os.path.join(data_path, file)) as data:
        for key in data.files:
            arr = data[key]
            if key == "frames":
                arr_moved = np.moveaxis(arr, 1, 3)
                arr_reshaped = arr_moved.reshape(arr.shape[0], 120, 160, 18)
                merged_frames.append(arr_reshaped)
                total_samples += arr.shape[0]
            elif key == "speeds":
                merged_speeds.append(arr)
            else:
                merged_actions.append(arr)

merged_frames = np.concatenate(merged_frames, axis=0, dtype=np.uint8)
merged_speeds = np.concatenate(merged_speeds, axis=0)
merged_actions = np.concatenate(merged_actions, axis=0)

print(merged_frames.shape)
print(merged_speeds.shape)
print(merged_actions.shape)

arr_dict = {
    "frames": merged_frames,
    "speeds": merged_speeds,
    "actions": merged_actions,
}

for file in files:
    os.remove(os.path.join(data_path, file))

np.savez("bc_data/merged.npz", **arr_dict)

gc.collect()

