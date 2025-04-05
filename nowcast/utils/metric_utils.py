import numpy as np
from nowcast.config import HEMConfig, TFDataConfig
from nowcast.utils.normalize import hem_denormalize
from skimage.metrics import structural_similarity as ssim


def mse_frame(y_true, y_pred):
    denorm_true = hem_denormalize(y_true)
    denorm_pred = hem_denormalize(y_pred)
    batch_metric = [0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
    for batch_true, batch_pred in zip(denorm_true, denorm_pred):
        for i, (true_frame, pred_frame) in enumerate(zip(batch_true, batch_pred)):
            batch_metric[i] += ((true_frame - pred_frame) ** 2).mean()
    return batch_metric


def mae_frame(y_true, y_pred):
    denorm_true = hem_denormalize(y_true)
    denorm_pred = hem_denormalize(y_pred)
    batch_metric = [0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
    for batch_true, batch_pred in zip(denorm_true, denorm_pred):
        for i, (true_frame, pred_frame) in enumerate(zip(batch_true, batch_pred)):
            batch_metric[i] += np.abs(true_frame - pred_frame).mean()
    return batch_metric


def ssim_frame(y_true, y_pred):
    denorm_true = hem_denormalize(y_true)
    denorm_pred = hem_denormalize(y_pred)
    batch_metric = [0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
    for batch_true, batch_pred in zip(denorm_true, denorm_pred):
        for i, (true_frame, pred_frame) in enumerate(zip(batch_true, batch_pred)):
            batch_metric[i] += ssim(
                true_frame,
                pred_frame,
                data_range=HEMConfig.MAX - HEMConfig.MIN,
                channel_axis=-1,
            )  # type: ignore
    return batch_metric


def psnr_frame(y_true, y_pred):
    denorm_true = hem_denormalize(y_true)
    denorm_pred = hem_denormalize(y_pred)
    batch_metric = [0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
    for batch_true, batch_pred in zip(denorm_true, denorm_pred):
        for i, (true_frame, pred_frame) in enumerate(zip(batch_true, batch_pred)):
            mse = ((true_frame - pred_frame) ** 2).mean()
            data_range = HEMConfig.MAX - HEMConfig.MIN
            batch_metric[i] += (
                20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else 100
            )
    return batch_metric


def corrcoef_frame(y_true, y_pred):
    denorm_true = hem_denormalize(y_true)
    denorm_pred = hem_denormalize(y_pred)
    batch_metric = [0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
    for batch_true, batch_pred in zip(denorm_true, denorm_pred):
        for i, (true_frame, pred_frame) in enumerate(zip(batch_true, batch_pred)):
            true_flat = true_frame.flatten()
            pred_flat = pred_frame.flatten()

            # Check if either array has zero standard deviation
            std_true = np.std(true_flat)
            std_pred = np.std(pred_flat)

            if std_true == 0 or std_pred == 0:
                # If both arrays are identical constants, correlation is 1
                # Otherwise, correlation is 0
                if (
                    std_true == 0
                    and std_pred == 0
                    and np.array_equal(true_flat, pred_flat)
                ):
                    batch_metric[i] += 1.0
                else:
                    batch_metric[i] += 0.0
            else:
                batch_metric[i] += np.corrcoef(true_flat, pred_flat)[0, 1]
    return batch_metric


def rmse_frame(y_true, y_pred):
    denorm_true = hem_denormalize(y_true)
    denorm_pred = hem_denormalize(y_pred)
    batch_metric = [0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
    for batch_true, batch_pred in zip(denorm_true, denorm_pred):
        for i, (true_frame, pred_frame) in enumerate(zip(batch_true, batch_pred)):
            batch_metric[i] += np.sqrt(((true_frame - pred_frame) ** 2).mean())
    return batch_metric


# all these functions take a batch of windows
# and calculate a frame-by-frame metric.
# the batch_metric gets sent to the caller
# which calculates and keeps track of a batched running mean.
METRIC_FUNC_MAP = {
    "mse": mse_frame,
    "rmse": rmse_frame,
    "mae": mae_frame,
    "ssim": ssim_frame,
    "psnr": psnr_frame,
    "corrcoef": corrcoef_frame,
}


def flow_mse_frame(y_true, y_pred):
    window_metric = []
    for true_frame, pred_frame in zip(y_true, y_pred):
        mse = np.nanmean((true_frame - pred_frame) ** 2)
        if np.isnan(mse).any():
            mse = None

        window_metric.append(mse)
    return window_metric


def flow_rmse_frame(y_true, y_pred):
    window_metric = []
    for true_frame, pred_frame in zip(y_true, y_pred):
        rmse = np.sqrt(np.nanmean((true_frame - pred_frame) ** 2))
        if np.isnan(rmse).any():
            rmse = None

        window_metric.append(rmse)
    return window_metric


def flow_mae_frame(y_true, y_pred):
    window_metric = []
    for true_frame, pred_frame in zip(y_true, y_pred):
        mse = np.nanmean(np.abs(true_frame - pred_frame))
        if np.isnan(mse).any():
            mse = None

        window_metric.append(mse)
    return window_metric


def flow_psnr_frame(y_true, y_pred):
    data_range = HEMConfig.MAX - HEMConfig.MIN
    return [
        (
            None
            if mse is None
            else (20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else 100)
        )
        for mse in flow_mse_frame(y_true, y_pred)
    ]


# TODO: write ssim from scratch to take care of the nan values
#       check if you can update it for the model too
def flow_ssim_frame(y_true, y_pred):
    window_metric = [0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
    for i, (true_frame, pred_frame) in enumerate(zip(y_true, y_pred)):
        window_metric[i] += ssim(
            true_frame,
            pred_frame,
            data_range=HEMConfig.MAX - HEMConfig.MIN,
            channel_axis=-1,
        )  # type: ignore
    return window_metric


# TODO: fix nan values in the frames
def flow_corrcoef_frame(y_true, y_pred):
    window_metric = [0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
    for i, (true_frame, pred_frame) in enumerate(zip(y_true, y_pred)):
        true_flat = true_frame.flatten()
        pred_flat = pred_frame.flatten()

        # Check if either array has zero standard deviation
        std_true = np.nanstd(true_flat)
        std_pred = np.nanstd(pred_flat)

        if std_true == 0 or std_pred == 0:
            # If both arrays are identical constants, correlation is 1
            # Otherwise, correlation is 0
            if std_true == 0 and std_pred == 0 and np.array_equal(true_flat, pred_flat):
                window_metric[i] += 1.0
            else:
                window_metric[i] += 0.0
        else:
            window_metric[i] += np.corrcoef(true_flat, pred_flat)[0, 1]
    return window_metric


FLOW_METRIC_FUNC_MAP = {
    "mse": flow_mse_frame,
    "rmse": flow_rmse_frame,
    "mae": flow_mae_frame,
    "ssim": flow_ssim_frame,
    "psnr": flow_psnr_frame,
    "corrcoef": flow_corrcoef_frame,
}
