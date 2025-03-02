import keras.api as K
import tensorflow as tf

from .normalize import hem_denormalize


@K.utils.register_keras_serializable()
def denorm_rmse(y_true, y_pred):
    """
    This is only applicable for the `y` window.
    Therefore this only runs for the HEM window.

    Take the RMSE of the denormalized values.
    """
    return tf.sqrt(
        tf.reduce_mean(tf.square(hem_denormalize(y_true) - hem_denormalize(y_pred)))
    )


@K.utils.register_keras_serializable()
def non_zero_denorm_rmse(y_true, y_pred):
    """
    This is only applicable for the `y` window.
    Therefore this only runs for the HEM window.

    Take the RMSE of only the non-zero values, the RMSE is taken after denormalization.
    """
    mask = y_true < 1e-6
    return tf.sqrt(
        tf.reduce_mean(
            tf.square((hem_denormalize(y_true) - hem_denormalize(y_pred))[mask])
        )
    )


@K.utils.register_keras_serializable()
def weighted_denorm_rmse(y_true, y_pred):
    """
    This is only applicable for the `y` window.
    Therefore this only runs for the HEM window.

    Take the RMSE of the non-zero values and add a weight to the zero values.
    Again, only done after denormalization. The current weight distribution is
    75% for non-zero values and 25% for zero values.
    """
    mask = y_true < 1e-6
    return 0.75 * tf.sqrt(
        tf.reduce_mean(
            tf.square((hem_denormalize(y_true) - hem_denormalize(y_pred))[mask])
        )
    ) + 0.25 * tf.sqrt(
        tf.reduce_mean(
            tf.square((hem_denormalize(y_true) - hem_denormalize(y_pred))[~mask])
        )
    )
