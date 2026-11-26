import os
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import torch

from src.experiments.lane_segmentation_ppo.lane_segmentation_fine_tune.model import TwinLiteNetPlus
from src.experiments.lane_segmentation_ppo.lane_segmentation_fine_tune.dataset import IMG_H, IMG_W, PAD
from src.experiments.lane_segmentation_ppo.lane_segmentation_fine_tune.augment import PAD_VALUE

class LaneSegmentation:

    def __init__(self, weights="checkpoints/lane_seg/small.pth", config="small", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = TwinLiteNetPlus(SimpleNamespace(config=config))
        state = torch.load(weights, map_location=self.device, weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def segment(self, obs):
        if obs.ndim == 3:
            obs = obs[np.newaxis]

        frames = []
        for f in obs:
            if f.shape[:2] != (IMG_H, IMG_W):
                f = cv.resize(f, (IMG_W, IMG_H), interpolation=cv.INTER_LINEAR)
            frames.append(cv.copyMakeBorder(f, PAD, PAD, 0, 0, cv.BORDER_CONSTANT, value=PAD_VALUE))

        x = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2).copy()).to(self.device).float() / 255.0
        da_logits, ll_logits = self.model(x)
        da_mask = da_logits[:, :, PAD:-PAD].argmax(1)[0].byte().cpu().numpy()
        ll_mask = ll_logits[:, :, PAD:-PAD].argmax(1)[0].byte().cpu().numpy()
        return da_mask, ll_mask