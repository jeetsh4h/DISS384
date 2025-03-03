import numpy as np
from scipy.interpolate import NearestNDInterpolator

from ..config import OLRConfig, HEMConfig


def olr_normalize(olr):
    """
    Min-Max scaling for OLR values
    """
    return (olr - OLRConfig.MIN) / (OLRConfig.MAX - OLRConfig.MIN)


def olr_denormalize(olr):
    """
    Denormalize OLR values
    """
    return olr * (OLRConfig.MAX - OLRConfig.MIN) + OLRConfig.MIN


def hem_normalize(hem):
    """
    Min-Max scaling for HEM values
    """
    return (hem - HEMConfig.MIN) / (HEMConfig.MAX - HEMConfig.MIN)


def hem_denormalize(hem):
    """
    Denormalize HEM values
    """
    return hem * (HEMConfig.MAX - HEMConfig.MIN) + HEMConfig.MIN


def olr_window_normalize(olr_window):
    """
    For each frame in the olr_window (shape: [batch_size, 400, 400]):
        - Fill NaN values based on the closest neighbor
        - Clip values between OLRConfig.MIN and OLRConfig.MAX
    """
    for i in range(len(olr_window)):
        # Fill NaNs using the nearest neighbor approach
        frame_filled = _fill_nans_with_interpolation(olr_window[i])

        # Clip the frame within the allowable range
        olr_window[i] = np.clip(frame_filled, OLRConfig.MIN, OLRConfig.MAX)

    return olr_normalize(np.stack(olr_window).astype(np.float16)[..., np.newaxis])


def hem_window_normalize(hem_window):
    """
    For each frame in the hem_window (shape: [batch_size, 400, 400]):
        - Fill NaN values based on the closest neighbor
        - Clip values between HEMConfig.MIN and HEMConfig.MAX
    """

    for i in range(len(hem_window)):
        # Fill NaNs using the nearest neighbor approach
        frame_filled = _fill_nans_with_interpolation(hem_window[i])

        # Clip the frame within the allowable range
        hem_window[i] = np.clip(frame_filled, HEMConfig.MIN, HEMConfig.MAX)

    return hem_normalize(np.stack(hem_window).astype(np.float16)[..., np.newaxis])


def _fill_nans_with_interpolation(arr):
    """
    Fill NaN values in a 2D array using the nearest non-NaN neighbor.
    If interpolation fails for some values, fill with mean as fallback.
    """
    mask = np.isnan(arr)
    if not mask.any():
        return arr

    # First attempt: nearest neighbor interpolation
    x_non_nan, y_non_nan = np.where(~mask)
    if len(x_non_nan) == 0:  # All values are NaN
        return np.zeros_like(arr)  # Return zeros as fallback

    values_non_nan = arr[x_non_nan, y_non_nan]

    interp_nn = NearestNDInterpolator(
        np.array([x_non_nan, y_non_nan]).T, values_non_nan
    )

    x_nan, y_nan = np.where(mask)
    imputed_values = interp_nn(np.array([x_nan, y_nan]).T)
    arr[x_nan, y_nan] = imputed_values

    # Check if any NaNs remain and fill them with mean
    remaining_mask = np.isnan(arr)
    if remaining_mask.any():
        if len(values_non_nan) > 0:
            fill_value = np.mean(values_non_nan)
        else:
            raise ValueError("Too many NaN values")
        arr[remaining_mask] = fill_value

    assert not np.isnan(arr).any(), "NaN values still present after interpolation"

    return arr
