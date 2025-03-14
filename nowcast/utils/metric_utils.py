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


#  TODO: fix the nan issue
def corrcoef_frame(y_true, y_pred):
    denorm_true = hem_denormalize(y_true)
    denorm_pred = hem_denormalize(y_pred)
    batch_metric = [0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
    for batch_true, batch_pred in zip(denorm_true, denorm_pred):
        for i, (true_frame, pred_frame) in enumerate(zip(batch_true, batch_pred)):
            true_flat = true_frame.flatten()
            pred_flat = pred_frame.flatten()
            batch_metric[i] += np.corrcoef(true_flat, pred_flat)[0, 1]
    return batch_metric


def rmse_frame(y_true, y_pred):
    mse_metrics = mse_frame(y_true, y_pred)
    return [metric**0.5 for metric in mse_metrics]


METRIC_FUNC_MAP = {
    "mse": mse_frame,
    "rmse": rmse_frame,
    "mae": mae_frame,
    "ssim": ssim_frame,
    "psnr": psnr_frame,
    "corrcoef": corrcoef_frame,
}
