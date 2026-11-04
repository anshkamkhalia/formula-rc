import tensorflow as tf

def build_rl_driver(timesteps=6):

    vision_input = tf.keras.Input(shape=(120, 160, 3 * timesteps))
    speed_input = tf.keras.Input(shape=(1,)) # just speed in m/s

    speed_features = tf.keras.layers.Dense(32, activation="relu")(speed_input)

    x = tf.keras.layers.Conv2D(32, (5,5), strides=2, padding='same', activation='relu')(vision_input)
    x = tf.keras.layers.Conv2D(64, (5,5), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu')(x)
    encoder_out = tf.keras.layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu')(x)

    flattened = tf.keras.layers.Flatten()(encoder_out)

    merged_features = tf.keras.layers.Concatenate()([flattened, speed_features]) # concatenate

    x = tf.keras.layers.Dense(32, activation="relu")(merged_features)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # steering distribution
    steering_mu = tf.keras.layers.Dense(1, activation=None, name="steering_mu")(x)
    steering_std_raw = tf.keras.layers.Dense(1, activation=None, name="steering_std_raw")(x)

    # throttle distribution
    throttle_mu = tf.keras.layers.Dense(1, activation=None, name="throttle_mu")(x)
    throttle_std_raw = tf.keras.layers.Dense(1, activation=None, name="throttle_std_raw")(x)

    # critic dense
    x = tf.keras.layers.Dense(32, activation="relu")(flattened)

    # value for ppo
    value = tf.keras.layers.Dense(1, activation=None, name="value")(x)

    return tf.keras.Model(
        [vision_input, speed_input],
        [steering_mu, steering_std_raw, throttle_mu, throttle_std_raw, value]
    )