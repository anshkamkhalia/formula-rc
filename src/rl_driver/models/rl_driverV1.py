import tensorflow as tf

def build_rl_driver(timesteps=6):

    input_layer = tf.keras.Input(shape=(120, 160, 3 * timesteps))

    x = tf.keras.layers.Conv2D(32, (5,5), strides=2, padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(64, (5,5), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # steering distribution
    steering_mu = tf.keras.layers.Dense(1, activation=None)(x)
    steering_std_raw = tf.keras.layers.Dense(1, activation=None)(x)

    # throttle distribution
    throttle_mu = tf.keras.layers.Dense(1, activation=None)(x)
    throttle_std_raw = tf.keras.layers.Dense(1, activation=None)(x)

    return tf.keras.Model(
        input_layer,
        [steering_mu, steering_std_raw, throttle_mu, throttle_std_raw]
    )