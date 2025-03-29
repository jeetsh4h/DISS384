import keras.api as K

from ..config import TFDataConfig, MOSDACConfig


def encoder_decoder(
    pretrained_weights=None,
    input_size=(TFDataConfig.OLR_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1),
):
    inp = K.layers.Input(input_size)

    # ENCODER
    # 1st ConvLSTM layer
    x = K.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(inp)
    x = K.layers.LeakyReLU()(x)
    skip1 = x  # Store for skip connection
    x = K.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

    # 2nd ConvLSTM layer
    x = K.layers.ConvLSTM2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(x)
    x = K.layers.LeakyReLU()(x)
    skip2 = x  # Store for skip connection
    x = K.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

    # 3rd ConvLSTM layer
    x = K.layers.ConvLSTM2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(x)
    x = K.layers.LeakyReLU()(x)
    skip3 = x  # Store for skip connection
    x = K.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

    # DECODER
    # 1st ConvLSTM layer
    x = K.layers.ConvLSTM2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(x)
    x = K.layers.LeakyReLU()(x)
    x = K.layers.UpSampling3D(size=(1, 2, 2))(x)
    # Add skip connection
    x = K.layers.Concatenate(axis=-1)([x, skip3])

    # 2nd ConvLSTM layer
    x = K.layers.ConvLSTM2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(x)
    x = K.layers.LeakyReLU()(x)
    x = K.layers.UpSampling3D(size=(1, 2, 2))(x)
    # Add skip connection
    x = K.layers.Concatenate(axis=-1)([x, skip2])

    # 3rd ConvLSTM layer
    x = K.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(x)
    x = K.layers.LeakyReLU()(x)
    x = K.layers.UpSampling3D(size=(1, 2, 2))(x)
    # Add skip connection
    x = K.layers.Concatenate(axis=-1)([x, skip1])

    # Final layers
    x = K.layers.Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="relu",
        data_format="channels_last",
    )(x)

    x = K.layers.Conv3D(
        filters=16,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="relu",
        data_format="channels_last",
    )(x)

    x = K.layers.Conv3D(
        filters=1,
        kernel_size=(1, 1, 1),
        padding="same",
        activation="sigmoid",
        data_format="channels_last",
    )(x)

    model = K.Model(inp, x)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model
