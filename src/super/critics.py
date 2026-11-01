import tensorflow as tf

def build_critic(state_shape, uses_latents):

    s_in = tf.keras.Input(shape=state_shape)
    a_in = tf.keras.Input(shape=(2,))

    if uses_latents:
        x = tf.keras.layers.Flatten()(s_in)

    else:
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 5, strides=2, activation="relu"))(s_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, 3, strides=2, activation="relu"))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, 3, strides=2, activation="relu"))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
        x = tf.keras.layers.Flatten()(x)
        
    x = tf.keras.layers.Concatenate()([x, a_in])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    q = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model([s_in, a_in], q)