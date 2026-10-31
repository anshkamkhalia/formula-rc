import tensorflow as tf
import numpy as np

class VAE(tf.keras.Model):

    def __init__(self, latent_dim=64, img_shape=(84, 128, 3)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.encoder = self.build_encoder()
        self.mu = tf.keras.layers.Dense(latent_dim)
        self.logvar = tf.keras.layers.Dense(latent_dim)
        self.decoder = self.build_decoder()

    def build_encoder(self):
        inputs = tf.keras.Input(shape=(84, 128, 3)) # cropped image 

        # conv block 1
        x = tf.keras.layers.Conv2D(32, (3,3), strides=2, padding="same")(inputs) # (42, 64, 32)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x) # silu activation

        # conv block 2
        x = tf.keras.layers.Conv2D(64, (5,5), strides=2, padding="same")(x) # (21, 32, 64)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

        # conv block 3 (no downsampling)
        x = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding="same")(x) # (21, 32)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

        self.feat_shape = tuple(x.shape[1:]) 
        out = tf.keras.layers.Flatten()(x) # 43008

        return tf.keras.Model(inputs, out, name="encoder_backbone")

    def build_decoder(self):

        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(int(np.prod(self.feat_shape)), activation="swish")(latent_inputs)
        x = tf.keras.layers.Reshape(self.feat_shape)(x)

        x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)

        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)

        # no upsampling here
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)

        x = tf.keras.layers.Conv2D(3, 3, strides=1, padding="same")(x)

        outputs = tf.keras.layers.Activation("sigmoid")(x)

        return tf.keras.Model(latent_inputs, outputs, name="decoder")

    def encode(self, x, training=False):
        h = self.encoder(x, training=training)
        mu = self.mu(h)
        logvar = tf.clip_by_value(self.logvar(h), -10.0, 10.0)
        return mu, logvar

    def call(self, x, training=False):
        mu, logvar = self.encode(x, training=training)
        z = mu + tf.math.exp(0.5 * logvar) * tf.random.normal(tf.shape(mu))
        recon = self.decoder(z, training=training)
        return recon, mu, logvar