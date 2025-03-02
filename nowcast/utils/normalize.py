import numpy as np
from scipy.ndimage import distance_transform_edt

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
        frame_filled = _fill_nans_with_nearest(olr_window[i])

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
        frame_filled = _fill_nans_with_nearest(hem_window[i])

        # Clip the frame within the allowable range
        hem_window[i] = np.clip(frame_filled, HEMConfig.MIN, HEMConfig.MAX)

    return hem_normalize(np.stack(hem_window).astype(np.float16)[..., np.newaxis])


def _fill_nans_with_nearest(arr):
    """
    Fill NaN values in a 2D array using the nearest non-NaN neighbor.
    """
    mask = np.isnan(arr)
    if not mask.any():
        return arr

    # Get indices of the nearest valid value for every element
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(indices)]  # type: ignore
