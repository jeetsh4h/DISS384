import keras.api as K

from ..config import MOSDACConfig, TrainConfig


def encoder_decoder(
    pretrained_weights=None,
    input_size=(None, TrainConfig.OLR_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1),
):
    inp = K.layers.Input(input_size)

    ########## encoder ##########

    # 1st ConvLSTM layer
    x = K.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(inp)
    x = K.layers.LeakyReLU()(x)
    x = K.layers.BatchNormalization()(x)

    # Store the output for skip connection
    skip_connection_1 = x

    x = K.layers.MaxPooling3D(pool_size=(1, 2, 2))(
        x
    )  # Temporal dimension remains unchanged

    # 2nd ConvLSTM layer
    x = K.layers.ConvLSTM2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(x)
    x = K.layers.LeakyReLU()(x)
    x = K.layers.MaxPooling3D(pool_size=(1, 2, 2))(
        x
    )  # Temporal dimension remains unchanged

    ########## decoder ##########

    # 1st ConvLSTM layer in the decoder
    x = K.layers.ConvLSTM2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(x)
    x = K.layers.LeakyReLU()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.UpSampling3D(size=(1, 2, 2))(x)

    # 2nd ConvLSTM layer in the decoder
    x = K.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        data_format="channels_last",
    )(x)
    x = K.layers.LeakyReLU()(x)
    x = K.layers.UpSampling3D(size=(1, 2, 2))(x)

    # Add skip connection from the first encoder layer
    x = K.layers.Concatenate()([skip_connection_1, x])

    # Final Layers
    x = K.layers.Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="sigmoid",
        data_format="channels_last",
    )(x)

    x = K.layers.Conv3D(
        filters=16,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="sigmoid",
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
