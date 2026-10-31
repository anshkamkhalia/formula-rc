import os
import numpy as np
import tensorflow as tf
import cv2 as cv
from src.rl_driver.v5.vae.vae import VAE
from keras.applications.vgg16 import VGG16, preprocess_input

# configs (must match training so the val split and sample picks are identical)
DATA_PATH = "vae_data/merged/merged.npy"
WEIGHTS_PATH = "checkpoints/vae.weights.h5"
VAL_FRAC = 0.05
N_RECON_SAMPLES = 8
RECON_DIR = "vae_recons"
os.makedirs(RECON_DIR, exist_ok=True)

# load data (memory-mapped so we don't pull all 700MB into RAM)
data = np.load(DATA_PATH, mmap_mode="r")

# reproduce the exact same split + sample selection as training
rng = np.random.default_rng(42)
idx = rng.permutation(len(data))
n_val = int(len(data) * VAL_FRAC)
val_data = data[idx[:n_val]]
recon_idx = rng.choice(len(val_data), size=N_RECON_SAMPLES, replace=False)
del data

def to_float(x):
    return tf.cast(x, tf.float32) / 255.0

# build the model with a dummy forward pass before loading weights
vae = VAE()
vae(to_float(val_data[:1]), training=False)
vae.load_weights(WEIGHTS_PATH)

def log_reconstructions(tag="test"):

    orig = val_data[recon_idx]
    mu, logvar = vae.encode(to_float(orig), training=False)
    recon = vae.decoder(mu, training=False)
    recon = np.clip(recon.numpy() * 255.0, 0, 255).astype(np.uint8)

    grid = np.concatenate([np.concatenate(list(orig), axis=1),
                           np.concatenate(list(recon), axis=1)], axis=0)

    path = os.path.join(RECON_DIR, f"{tag}.png")
    grid = cv.cvtColor(grid, cv.COLOR_BGR2RGB)
    cv.imwrite(path, grid)
    print(f"saved reconstructions to {path}")

if __name__ == "__main__":
    log_reconstructions("best_checkpoint")

vgg = VGG16(weights="imagenet", include_top=False, input_shape=(84, 128, 3))
PERC_LAYERS = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"]
feature_extractor = tf.keras.Model(
    vgg.input, [vgg.get_layer(n).output for n in PERC_LAYERS])

def perceptual_loss(x, recon):
    fx = feature_extractor(preprocess_input(x * 255), training=False)
    fr = feature_extractor(preprocess_input(recon * 255), training=False)
    total = 0.0
    for a, b in zip(fx, fr):
        a = tf.math.l2_normalize(a, axis=-1, epsilon=1e-7)
        b = tf.math.l2_normalize(b, axis=-1, epsilon=1e-7)
        total += tf.reduce_mean(tf.reduce_sum(tf.square(a - b), axis=-1), axis=[1, 2])
    return total

x = to_float(val_data[:128])
recon, mu, logvar = vae(x, training=False)
recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - recon), axis=[1, 2, 3]))
perc_raw = tf.reduce_mean(perceptual_loss(x, recon))

print(f"recon {recon_loss:.2f}, raw perc {perc_raw:.4f}")
print(f"suggested lambda: {0.2 * recon_loss / perc_raw:.1f}")