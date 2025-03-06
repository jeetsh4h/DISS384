import keras.api as K
import tensorflow as tf

from .normalize import hem_denormalize
from ..config import TFDataConfig, MOSDACConfig


@K.utils.register_keras_serializable()
def denorm_rmse(y_true, y_pred):
    """
    This is only applicable for the `y` window.
    Therefore this only runs for the HEM window.

    Take the RMSE of the denormalized values.
    """

    y_true_denorm = hem_denormalize(y_true)
    y_pred_denorm = hem_denormalize(y_pred)

    diff = y_true_denorm - y_pred_denorm

    return tf.sqrt(tf.reduce_mean(tf.square(diff)))


@K.utils.register_keras_serializable()
def non_zero_denorm_rmse(y_true, y_pred):
    """
    This is only applicable for the `y` window.
    Therefore this only runs for the HEM window.

    Take the RMSE of only the non-zero values, the RMSE is taken after denormalization.
    """

    y_true_denorm = hem_denormalize(y_true)
    y_pred_denorm = hem_denormalize(y_pred)

    diff = y_true_denorm - y_pred_denorm

    non_zero_mask = y_true_denorm > TFDataConfig.TOLERANCE

    return tf.sqrt(tf.reduce_mean(tf.square(diff[non_zero_mask])))


@K.utils.register_keras_serializable()
def weighted_denorm_rmse(y_true, y_pred):
    """
    This is only applicable for the `y` window.
    Therefore this only runs for the HEM window.

    Take the RMSE of the non-zero values and add a weight to the zero values.
    Again, only done after denormalization. The current weight distribution is
    80% for non-zero values and 20% for zero values.
    """
    y_true_denorm = hem_denormalize(y_true)
    y_pred_denorm = hem_denormalize(y_pred)

    non_zero_mask = y_true_denorm > TFDataConfig.TOLERANCE

    diff = y_true_denorm - y_pred_denorm

    non_zero_rmse = tf.sqrt(tf.reduce_mean(tf.square(diff[non_zero_mask])))
    zero_rmse = tf.sqrt(tf.reduce_mean(tf.square(diff[~non_zero_mask])))

    return (0.80 * non_zero_rmse) + (0.20 * zero_rmse)


@K.utils.register_keras_serializable()
def weighted_pixel_loss(y_true, y_pred):
    y_true_denorm = hem_denormalize(y_true)
    y_pred_denorm = hem_denormalize(y_pred)

    diff = y_true_denorm - y_pred_denorm

    # Create weights based on the true rainfall intensity
    # Higher rainfall values get exponentially higher weights
    intensity_weights = tf.pow(y_true_denorm, 2.0)

    # TODO: check if removing the root works better or worse?
    root_squared_error = tf.sqrt(tf.square(diff))

    weighted_squared_error = intensity_weights * root_squared_error

    return tf.reduce_mean(weighted_squared_error)


@K.utils.register_keras_serializable()
def frame_loss(y_true, y_pred):
    total_loss = 0.0

    # Process each frame in the sequence without unstacking batches
    for y_true_batched_frame, y_pred_batched_frame in zip(
        tf.unstack(y_true, axis=1), tf.unstack(y_pred, axis=1)  # type:ignore
    ):
        total_loss += weighted_pixel_loss(y_true_batched_frame, y_pred_batched_frame)

    return total_loss


################# TODO #################


# TODO: fix perceptual loss and combined loss
@K.utils.register_keras_serializable()
def perceptual_loss(y_true, y_pred):
    xception_base = K.applications.Xception(
        False, input_shape=(*MOSDACConfig.FRAME_SIZE, 3)
    )

    feature_extractor = K.models.Model(
        inputs=xception_base.input,
        outputs=xception_base.get_layer("block12_sepconv1_act").output,
    )
    feature_extractor.trainable = False

    features_true = feature_extractor(tf.concat([y_true, y_true, y_true], axis=-1))
    features_pred = feature_extractor(tf.concat([y_pred, y_pred, y_pred], axis=-1))

    # RMSE of features
    return tf.sqrt(K.losses.mean_squared_error(features_true, features_pred))


@K.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    # Initialize the total loss
    total_loss = 0.0

    # Alpha parameter to weight between pixel loss and perceptual loss
    alpha = 0.7  # Adjust this value as needed

    # Process each frame in the sequence without unstacking batches
    for y_true_batched_frame, y_pred_batched_frame in zip(
        tf.unstack(y_true, axis=1), tf.unstack(y_pred, axis=1)  # type:ignore
    ):
        frame_pixel_loss = weighted_pixel_loss(
            y_true_batched_frame, y_pred_batched_frame
        )
        frame_perceptual_loss = perceptual_loss(
            y_true_batched_frame, y_pred_batched_frame
        )

        # Combine losses for this frame
        frame_loss = alpha * frame_pixel_loss + (1 - alpha) * frame_perceptual_loss

        # Add to total loss (sum, not average)
        total_loss += frame_loss

    return total_loss
