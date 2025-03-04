import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from ..config import TrainConfig, DataConfig, VizConfig
from ..utils.file_utils import find_by_date
from ..utils.normalize import hem_window_normalize, olr_window_normalize


def window_by_date(date: dt.datetime, offset: int):
    olr_start_time = date
    olr_end_time = date + dt.timedelta(hours=TrainConfig.OLR_WINDOW_SIZE)

    hem_start_time = olr_end_time + dt.timedelta(hours=offset)
    hem_end_time = hem_start_time + dt.timedelta(hours=TrainConfig.HEM_WINDOW_SIZE)

    olr_frame_fns, olr_frame_ts = find_by_date(
        olr_start_time,
        olr_end_time,
        str(DataConfig.CACHE_DIR / "OLR"),
        DataConfig.NP_FILENAME_FMT.replace("%s", "OLR", 1),
        "npy",
        30,
        0,
        False,
    )

    hem_frame_fns, hem_frame_ts = find_by_date(
        hem_start_time,
        hem_end_time,
        str(DataConfig.CACHE_DIR / "HEM"),
        DataConfig.NP_FILENAME_FMT.replace("%s", "HEM", 1),
        "npy",
        30,
        0,
        False,
    )

    if (
        len(olr_frame_fns) < TrainConfig.OLR_WINDOW_SIZE
        or len(hem_frame_fns) < TrainConfig.HEM_WINDOW_SIZE
        or None in olr_frame_fns
        or None in hem_frame_fns
    ):
        print(f"Warning: Not enough data for date {date}")
        return None, None

    olr_norm_data = olr_window_normalize([np.load(fn) for fn in olr_frame_fns])  # type: ignore
    hem_norm_data = hem_window_normalize([np.load(fn) for fn in hem_frame_fns])  # type: ignore

    return olr_norm_data, hem_norm_data


def visualize_hem_compare(y_pred, y_true, date, offset):
    fig = plt.figure(figsize=(int((TrainConfig.HEM_WINDOW_SIZE * 4.5) + 2), 14))
    fig.suptitle(f"Prediction Comparison for {offset} hours offset")

    hem_pred, hem_true = fig.subfigures(2, 1, height_ratios=[1, 1])

    # Predicted HEM subfigure
    hem_pred.suptitle(f"Predicted HEM")
    axs_pred = hem_pred.subplots(1, TrainConfig.HEM_WINDOW_SIZE)
    im_pred = None
    for i, ax in enumerate(axs_pred):
        ax.set_title(
            dt.datetime.strftime(
                date
                + dt.timedelta(
                    hours=offset, minutes=30 * (TrainConfig.OLR_WINDOW_SIZE + i)
                ),
                "%H:%M %d-%m-%Y",
            )
        )
        ax.plot(*VizConfig.COAST_PLOT_PARAMS)
        im_pred = ax.imshow(y_pred[i, ...], **VizConfig.HEM_IMSHOW_KWARGS)
    assert im_pred is not None
    hem_pred.subplots_adjust(right=0.84)
    pred_cbar_ax = hem_pred.add_axes([0.85, 0.15, 0.008, 0.7])
    fig.colorbar(im_pred, cax=pred_cbar_ax)
    pred_cbar_ax.set_ylabel("HEM  (mm/hr)", rotation=270, labelpad=15)

    # True HEM subfigure
    hem_true.suptitle(f"True HEM")
    axs_true = hem_true.subplots(1, TrainConfig.HEM_WINDOW_SIZE)
    im_true = None
    for i, ax in enumerate(axs_true):
        ax.set_title(
            dt.datetime.strftime(
                date
                + dt.timedelta(
                    hours=offset, minutes=30 * (TrainConfig.OLR_WINDOW_SIZE + i)
                ),
                "%H:%M %d-%m-%Y",
            )
        )
        ax.plot(*VizConfig.COAST_PLOT_PARAMS)
        im_true = ax.imshow(y_true[i, ...], **VizConfig.HEM_IMSHOW_KWARGS)
    assert im_true is not None
    hem_true.subplots_adjust(right=0.84)
    true_cbar_ax = hem_true.add_axes([0.85, 0.15, 0.008, 0.7])
    fig.colorbar(im_true, cax=true_cbar_ax)
    true_cbar_ax.set_ylabel("HEM  (mm/hr)", rotation=270, labelpad=15)

    return fig
