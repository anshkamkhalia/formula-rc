import numpy as np
import tensorflow as tf
from src.rl_driver.v5.vae.vae import VAE
import math
import os
from tqdm import tqdm
import cv2 as cv
from keras.applications.vgg16 import VGG16, preprocess_input

# load vgg16 for perceptual loss
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(84, 128, 3))

PERC_LAYERS = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"] # feature extractor layers
LAMBDA_PERC = 18.0 # vgg perceptual loss dampening factor

vgg.trainable = False
feature_extractor = tf.keras.Model(
    vgg.input, [vgg.get_layer(name).output for name in PERC_LAYERS])
feature_extractor.trainable = False

data = np.load("vae_data/merged/merged.npy") # stays uint8, around 700MB

# configs
BUFFER_SIZE = len(data)
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
VAL_FRAC = 0.05
RECONSTRUCTION_LOG_EPOCHS = 3 # every n epochs
N_RECON_SAMPLES = 8
RECON_DIR = "vae_recons"
os.makedirs(RECON_DIR, exist_ok=True)
WARMUP_EPOCHS = 10
BETA_TARGET = 0.1

vae = VAE()
optimizer = tf.keras.optimizers.Adam(LR)
beta = tf.Variable(initial_value=0.0, name="scheduled_beta")
step_counter = 0

rng = np.random.default_rng(42)
idx = rng.permutation(len(data))
n_val = int(len(data) * VAL_FRAC)
val_data = data[idx[:n_val]]
train_data = data[idx[n_val:]]
del data

n_train_batches = math.ceil(len(train_data) / BATCH_SIZE)
n_val_batches = math.ceil(len(val_data) / BATCH_SIZE)
WARMUP_STEPS = WARMUP_EPOCHS * n_train_batches 

recon_idx = rng.choice(len(val_data), size=N_RECON_SAMPLES, replace=False)

def perceptual_loss(x, recon):

    # get vgg features
    x_in = preprocess_input(x * 255) # * 255 to convert 0-1 to 0-255
    recon_in = preprocess_input(recon * 255) # same here

    # use selected layers to run feature extractor
    feature_map_x = feature_extractor(x_in, training=False)
    feature_map_recon = feature_extractor(recon_in, training=False)
    total = 0.0

    for spatial_dim_x, spatial_dim_r in zip(feature_map_x, feature_map_recon):
        # unit normalize each spatial position (lpips style)
        spatial_dim_x = tf.math.l2_normalize(spatial_dim_x, axis=-1, epsilon=1e-7)
        spatial_dim_r = tf.math.l2_normalize(spatial_dim_r, axis=-1, epsilon=1e-7)
        d = tf.reduce_sum(tf.square(spatial_dim_x - spatial_dim_r), axis=-1)
        total += tf.reduce_mean(d, axis=[1,2])
    return total
        
def to_float(x):
    return tf.cast(x, tf.float32) / 255.0 # keep uint8 in ram, convert per batch

def loss_fn(x, recon, mu, logvar):

    # reconstruction: sum over pixels, mean over batch
    recon_loss = tf.reduce_sum(tf.square(x - recon), axis=[1, 2, 3])
    kl_loss = -0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)

    recon_loss = tf.reduce_mean(recon_loss)
    kl_loss = tf.reduce_mean(kl_loss)
    total = recon_loss + beta * kl_loss

    # vgg loss
    perc_loss = perceptual_loss(x, recon)
    perc_loss *= LAMBDA_PERC # lamba factor
    total += perc_loss

    return total, recon_loss, kl_loss, perc_loss

def log_reconstructions(epoch):

    orig = val_data[recon_idx]
    mu, logvar = vae.encode(to_float(orig), training=False)
    recon = vae.decoder(mu, training=False)  # decode the mean for a clean reconstruction
    recon = np.clip(recon.numpy() * 255.0, 0, 255).astype(np.uint8)

    # top row: originals, bottom row: reconstructions
    grid = np.concatenate([np.concatenate(list(orig), axis=1),
                           np.concatenate(list(recon), axis=1)], axis=0)

    path = os.path.join(RECON_DIR, f"epoch_{epoch+1:03d}.png")
    grid = cv.cvtColor(grid, cv.COLOR_BGR2RGB)
    cv.imwrite(path, grid)
    tqdm.write(f"saved reconstructions to {path}")

# training and val metric
train_loss = tf.keras.metrics.Mean()
train_recon = tf.keras.metrics.Mean()
train_kl = tf.keras.metrics.Mean()
val_loss = tf.keras.metrics.Mean()
val_recon = tf.keras.metrics.Mean()
val_kl = tf.keras.metrics.Mean()
train_perc = tf.keras.metrics.Mean()
val_perc = tf.keras.metrics.Mean()

# dataset is small so we can load it into ram
train_ds = (tf.data.Dataset.from_tensor_slices(train_data)
            
            .shuffle(len(train_data), reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices(val_data)
          
          .batch(BATCH_SIZE)
          .map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
          .prefetch(tf.data.AUTOTUNE))

@tf.function
def train_step(x):

    with tf.GradientTape() as tape:

        recon, mu, logvar = vae(x, training=True)
        loss, rl, kl, pl = loss_fn(x, recon, mu, logvar) # new perc loss

    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    n = tf.shape(x)[0]

    train_loss.update_state(loss, sample_weight=n)
    train_recon.update_state(rl, sample_weight=n)
    train_kl.update_state(kl, sample_weight=n)
    train_perc.update_state(pl, sample_weight=n)

    return rl

@tf.function
def val_step(x):

    recon, mu, logvar = vae(x, training=False)
    loss, rl, kl, pl = loss_fn(x, recon, mu, logvar)

    n = tf.shape(x)[0]
    val_loss.update_state(loss, sample_weight=n)
    val_recon.update_state(rl, sample_weight=n)
    val_kl.update_state(kl, sample_weight=n)
    val_perc.update_state(pl, sample_weight=n)

    return rl

best_val = float("inf")
global_step = 0

for epoch in range(EPOCHS):
    for m in (train_loss, train_recon, train_kl, train_perc,
            val_loss, val_recon, val_kl, val_perc):
        m.reset_state()

    pbar = tqdm(train_ds, total=n_train_batches, desc=f"epoch {epoch+1}/{EPOCHS}", unit="batch")

    for step, x_batch in enumerate(pbar):
        beta.assign(BETA_TARGET * min(1.0, global_step / WARMUP_STEPS))
        train_step(x_batch)
        global_step += 1

        if step % 10 == 0:
            pbar.set_postfix(loss=f"{train_loss.result():.1f}",
                             recon=f"{train_recon.result():.1f}",
                             kl=f"{train_kl.result():.1f}")

    for x_batch in tqdm(val_ds, total=n_val_batches, desc="  val", unit="batch", leave=False):
        rl = val_step(x_batch)

    # reconstruction logging
    if epoch % RECONSTRUCTION_LOG_EPOCHS == 0 or epoch == EPOCHS - 1:
        log_reconstructions(epoch)

    tqdm.write(f"epoch {epoch+1}/{EPOCHS} | "
               f"train {train_loss.result():.2f} (recon {train_recon.result():.2f}, kl {train_kl.result():.2f}, perc {train_perc.result():.2f}) | "
               f"val {val_loss.result():.2f} (recon {val_recon.result():.2f}, kl {val_kl.result():.2f}, perc {val_perc.result():.2f})")

    if val_loss.result() < best_val:
        best_val = float(val_loss.result())
        vae.save_weights("checkpoints/vae_best.weights.h5")