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
    intensity_weights = tf.maximum(tf.pow(y_true_denorm, 2.0), 1.0)

    # TODO: check if removing the root works better or worse?
    root_squared_error = tf.sqrt(tf.square(diff))

    weighted_squared_error = intensity_weights * root_squared_error

    return tf.reduce_mean(weighted_squared_error)


@K.utils.register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal Loss adapted for regression on satellite data.

    This puts more focus on hard examples (high error) and can weight by rainfall intensity.
    - gamma: focusing parameter, higher values increase focus on hard examples
    - alpha: weighting factor for rainfall intensity
    """
    y_true_denorm = hem_denormalize(y_true)
    y_pred_denorm = hem_denormalize(y_pred)

    # Calculate absolute error
    abs_error = tf.abs(y_true_denorm - y_pred_denorm)

    # Create modulating factor based on error (smaller errors get reduced weight)
    modulating_factor = tf.pow(abs_error, gamma)

    # Weight by rainfall intensity (optional)
    intensity_weights = tf.pow(y_true_denorm + 1.0, alpha)

    # Combine weights and errors
    weighted_errors = intensity_weights * modulating_factor * abs_error

    return tf.reduce_mean(weighted_errors)


@K.utils.register_keras_serializable()
class FocalLoss(K.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return focal_loss(y_true, y_pred, self.gamma, self.alpha)


@K.utils.register_keras_serializable()
class PerceptualLoss(K.losses.Loss):
    def __init__(self, name="perceptual_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        xception_base = K.applications.Xception(
            False, input_shape=(*MOSDACConfig.FRAME_SIZE, 3)
        )

        self.feature_extractor = K.models.Model(
            inputs=xception_base.input,
            outputs=xception_base.get_layer("block12_sepconv1_act").output,
        )
        self.feature_extractor.trainable = False

    def call(self, y_true, y_pred):
        features_true = self.feature_extractor(
            tf.concat([y_true, y_true, y_true], axis=-1)
        )
        features_pred = self.feature_extractor(
            tf.concat([y_pred, y_pred, y_pred], axis=-1)
        )

        # MSE between features
        return 5.0 * K.losses.mean_squared_error(features_true, features_pred)  # type: ignore


@K.utils.register_keras_serializable()
class CombinedLoss(K.losses.Loss):
    def __init__(
        self, name="test_loss", use_focal=False, gamma=2.0, alpha=0.25, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.perceptual_loss = PerceptualLoss()
        self.use_focal = use_focal
        self.focal_loss = FocalLoss(gamma, alpha)

    def call(self, y_true, y_pred):
        total_loss = 0.0

        # Process each frame in the sequence without unstacking batches
        for y_true_batched_frame, y_pred_batched_frame in zip(
            tf.unstack(y_true, axis=1), tf.unstack(y_pred, axis=1)  # type:ignore
        ):
            frame_perceptual_loss = self.perceptual_loss(
                y_true_batched_frame, y_pred_batched_frame
            )

            if self.use_focal:
                frame_pixel_loss = self.focal_loss(
                    y_true_batched_frame, y_pred_batched_frame
                )
                total_loss += frame_perceptual_loss + (0.01 * frame_pixel_loss)  # type: ignore
            else:
                frame_pixel_loss = weighted_denorm_rmse(
                    y_true_batched_frame, y_pred_batched_frame
                )
                total_loss += frame_perceptual_loss + (2.0 * frame_pixel_loss)  # type: ignore

        return total_loss
