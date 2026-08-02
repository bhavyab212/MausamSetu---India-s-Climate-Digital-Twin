import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Lambda
from tensorflow.keras.models import Model


def build_unet(seq_length, lat_dim, lon_dim, channels=2):
    """
    U-Net with time-flattened input. Skip connections are aligned via tf.image.resize
    (avoids Resizing(..., c2.shape[...]) which can be None / wrong on some Keras builds
    and has been observed to yield all-NaN outputs after load_weights on Colab).
    """
    inputs = Input(shape=(seq_length, lat_dim, lon_dim, channels))

    def flatten_time(x):
        s = tf.shape(x)
        return tf.reshape(x, [s[0], s[2], s[3], seq_length * channels])

    def resize_like(args):
        x, ref = args
        rh = tf.shape(ref)[1]
        rw = tf.shape(ref)[2]
        return tf.image.resize(x, [rh, rw], method="bilinear")

    x = Lambda(flatten_time)(inputs)

    # Encoder
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation="relu", padding="same")(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(p2)

    # Decoder — match skip tensor sizes dynamically
    u1 = UpSampling2D((2, 2))(c3)
    u1 = Lambda(resize_like)([u1, c2])
    concat1 = concatenate([u1, c2])
    c4 = Conv2D(32, (3, 3), activation="relu", padding="same")(concat1)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = Lambda(resize_like)([u2, c1])
    concat2 = concatenate([u2, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat2)

    # Match normalized targets [0, 1] (same as ConvLSTM / training pipeline).
    # Unclipped linear outputs often go <0 or >1 → broken denorm (negative rain, stripes).
    x_out = Conv2D(channels, (1, 1), activation="linear")(c5)
    outputs = Lambda(lambda t: tf.clip_by_value(t, 0.0, 1.0))(x_out)

    model = Model(inputs=inputs, outputs=outputs, name="UNet")
    return model
