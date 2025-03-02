"""
Data loader utilities for the nowcast package.
"""

import numpy as np
import datetime as dt

from ..utils.cache_utils import find_by_date
from ..config import DataConfig, TrainConfig
from ..utils.normalize import olr_window_normalize, hem_window_normalize


def load_data_generator(
    window_fns: list[list[str]],
    window_timestamps: list[dt.datetime],
    batch_size: int,
    shuffle: bool,
    yield_batch_ts: bool,
):
    """
    Data generator for loading OLR and HEM data for training and evaluation.

    Parameters
    ----------
    window_fns : list[list[str]]
        List of lists containing OLR and HEM frame filenames
    window_timestamps : list[dt.datetime]
        Timestamps corresponding to each window
    batch_size : int
        Number of samples in each batch
    shuffle : bool
        Whether to shuffle the data
    yield_batch_ts : bool
        Whether to yield timestamps along with data batches

    Yields
    ------
    tuple
        (x_batch, y_batch) or (x_batch, y_batch, timestamps)
        where x_batch is OLR input and y_batch is HEM target
    """
    while True:
        if shuffle:
            idx = np.random.permutation(len(window_fns))
        else:
            idx = np.arange(len(window_fns))

        for i in range(0, len(window_fns), batch_size):
            batch_ids = idx[i : i + batch_size]

            olr_batch = []
            hem_batch = []
            ts_batch = []

            for j in batch_ids:
                olr_window_fns = window_fns[j][: TrainConfig.OLR_WINDOW_SIZE]
                hem_window_fns = window_fns[j][TrainConfig.OLR_WINDOW_SIZE :]

                assert len(olr_window_fns) == TrainConfig.OLR_WINDOW_SIZE
                assert len(hem_window_fns) == TrainConfig.HEM_WINDOW_SIZE

                olr_batch.append([np.load(frame_fn) for frame_fn in olr_window_fns])
                hem_batch.append([np.load(frame_fn) for frame_fn in hem_window_fns])

                ts_batch.append(window_timestamps[j])

            x_batch = olr_window_normalize(olr_batch)
            y_batch = hem_window_normalize(hem_batch)

            if yield_batch_ts:
                yield x_batch, y_batch, ts_batch
            else:
                yield x_batch, y_batch


def create_windows(
    start_time: dt.datetime,
    end_time: dt.datetime,
    hem_nowcast_offset: int,
) -> tuple[list[list[str]], list[dt.datetime]]:
    """
    Create input-output window pairs for training the model.

    Parameters
    ----------
    olr_frame_fns : list[str]
        List of OLR frame filenames
    hem_frame_fns : list[str]
        List of HEM frame filenames
    frame_timestamps : list[dt.datetime]
        Timestamps corresponding to each frame
    hem_nowcast_offset : int
        Number of timesteps to offset the HEM nowcast

    Returns
    -------
    tuple
        (window_fns, window_timestamps)
        where window_fns is a list of (olr_files, hem_files) tuples
        and window_timestamps is a list of timestamps for each window
    """
    olr_frame_fns, hem_frame_fns, frame_timestamps = _corr_fn_ts(start_time, end_time)

    olr_window_size = TrainConfig.OLR_WINDOW_SIZE
    hem_window_size = TrainConfig.HEM_WINDOW_SIZE

    window_fns: list[list[str]] = []
    window_timestamps: list[dt.datetime] = []

    # We need enough frames to create a complete window
    for i in range(len(frame_timestamps)):
        olr_timestamps = frame_timestamps[i : i + olr_window_size]
        if not _check_ts_order(olr_timestamps):
            continue

        olr_files = olr_frame_fns[i : i + olr_window_size]

        # Start HEM frames after OLR window, offset by hem_nowcast_offset
        hem_start_idx = i + olr_window_size + hem_nowcast_offset
        hem_end_idx = hem_start_idx + hem_window_size

        # Check if we have enough frames left
        if hem_end_idx > len(hem_frame_fns):
            continue

        hem_timestamps = frame_timestamps[hem_start_idx:hem_end_idx]
        if not _check_ts_order(hem_timestamps):
            continue

        hem_files = hem_frame_fns[hem_start_idx:hem_end_idx]

        window_fns.append(olr_files + hem_files)
        window_timestamps.append(frame_timestamps[i])

    return window_fns, window_timestamps


def _corr_fn_ts(start_time: dt.datetime, end_time: dt.datetime):
    """
    Maps OLR and HEM filenames to timestamps (based on index).
    Only keep the files for which both OLR and HEM data is available (for that timestamp).
    """
    olr_frame_fns, olr_frame_ts = find_by_date(
        start_time,
        end_time,
        str(DataConfig.CACHE_DIR / "OLR"),
        DataConfig.NP_FILENAME_FMT.replace("%s", "OLR", 1),
        "npy",
        30,
        0,
        False,
    )
    hem_frame_fns, hem_frame_ts = find_by_date(
        start_time,
        end_time,
        str(DataConfig.CACHE_DIR / "HEM"),
        DataConfig.NP_FILENAME_FMT.replace("%s", "HEM", 1),
        "npy",
        30,
        0,
        False,
    )

    assert len(olr_frame_fns) == len(
        hem_frame_fns
    ), "Unequal number of timestamps for OLR and HEM"
    assert len(olr_frame_fns) == len(
        olr_frame_ts
    ), "Unequal number of timestamps for OLR and HEM"

    olr_fn: list[str] = []
    hem_fn: list[str] = []
    ts: list[dt.datetime] = []

    assert hem_frame_ts == olr_frame_ts, "Timestamps do not match for OLR and HEM"

    for olr_curr, hem_curr, olr_ts in zip(olr_frame_fns, hem_frame_fns, olr_frame_ts):
        if olr_curr is None or hem_curr is None:
            continue

        curr_ts = olr_ts
        olr_fn.append(olr_curr)
        hem_fn.append(hem_curr)
        ts.append(curr_ts)

    assert len(olr_fn) == len(hem_fn), "Unequal number of files for OLR and HEM"

    return olr_fn, hem_fn, ts


def _check_ts_order(ts: list[dt.datetime], timestep: int = 30, ascending: bool = True):
    """
    Check if timestamps are in increasing order.
    """
    if ascending:
        if not all(ts[i] < ts[i + 1] for i in range(len(ts) - 1)):
            return False
    else:
        if not all(ts[i] > ts[i + 1] for i in range(len(ts) - 1)):
            return False

    # Check if the difference between timestamps is equal to timestep
    if not all(
        (ts[i + 1] - ts[i]).seconds == timestep * 60 for i in range(len(ts) - 1)
    ):
        return False

    return True
