# Adapted from TwinLiteNetPlus (BDD100K.py):
# https://github.com/chequanghuy/TwinLiteNetPlus
#
# Copyright (c) 2025 Chế Quang Huy
# Licensed under the MIT License. See licenses/TwinLiteNet-LICENSE for full text.
#
# Modifications: RGB input instead of BGR, nearest-neighbor warping for masks,
# symmetric scale range, OpenCV blurs instead of scikit-image, added JPEG,
# noise and crop helpers.

import math
import random

import cv2
import numpy as np

PAD_VALUE = (114, 114, 114)


def letterbox(im, new_shape=(384, 640), color=PAD_VALUE):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def random_perspective(img, da, ll, degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0):
    height, width = img.shape[:2]

    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2

    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    M = T @ S @ R @ P @ C  # order matters (right to left)

    def warp(x, interp, border):
        if perspective:
            return cv2.warpPerspective(x, M, (width, height), flags=interp, borderValue=border)
        return cv2.warpAffine(x, M[:2], (width, height), flags=interp, borderValue=border)

    return (warp(img, cv2.INTER_LINEAR, PAD_VALUE),
            warp(da, cv2.INTER_NEAREST, 0),
            warp(ll, cv2.INTER_NEAREST, 0))


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_gaussian_blur(img, sigma_a=1.15, sigma_b=0.15):
    sigma = sigma_b + random.random() * sigma_a
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def random_bilateral_blur(img):
    return cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=random.uniform(1.0, 3.0))


def random_jpeg(img, quality_low=30, quality_high=90):
    q = random.randint(quality_low, quality_high)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def random_noise(img, sigma_max=8.0):
    noise = np.random.normal(0, random.uniform(0, sigma_max), img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_crop(img, da, ll, crop_h, crop_w):
    h, w = img.shape[:2]
    if crop_h >= h or crop_w >= w:
        return img, da, ll
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    return (img[y:y + crop_h, x:x + crop_w],
            da[y:y + crop_h, x:x + crop_w],
            ll[y:y + crop_h, x:x + crop_w])