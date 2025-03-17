import json
import keras.api as K
import datetime as dt
from pathlib import Path
from pprint import pprint

from ...utils.metric_utils import *
from ...config import TestConfig, TFDataConfig
from ...utils.data_loader import create_windows, load_data_generator


def setup_parser(subparsers):
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Calculate metrics for a model on a set of test data.",
    )

    metrics_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="The model to test, from the log directory only (for now).",
    )

    metrics_parser.add_argument(
        "--metric",
        "-me",
        nargs="+",
        type=str,
        choices=["rmse", "corrcoef", "ssim", "mse", "mae", "psnr", "all"],
        required=True,
        help="One or more metric names to compute.",
    )

    metrics_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for evaluating",
    )

    metrics_parser.add_argument(
        "--offset",
        "-no",
        type=int,
        help="The offset at which HEM is nowcasted from the input OLR. Will be taken from the model's directory name if not provided.",
    )

    return metrics_parser


def execute(args):
    model_dir: Path = TFDataConfig.TB_LOG_DIR / args.model
    if not model_dir.exists():
        print(
            f"Error: The specified model '{args.model}' does not exist in the log directory."
        )
        return 1

    model_path = model_dir / "model.keras"
    if not model_path.exists():
        print(
            f"Error: The specified model '{args.model}' does not contain a trained model."
        )
        return 1

    offset = None
    if args.offset is None:
        try:
            offset = int(args.model.split("_")[-1])
        except ValueError:
            raise ValueError(
                f"Error: The model directory name '{args.model}' does not contain an offset."
            )
    else:
        offset = args.offset

    assert offset is not None

    test_window_fns, test_window_timestamps = create_windows(
        TestConfig.TEST_START_DT,
        TestConfig.TEST_END_DT,
        offset,
    )
    # TODO: wrap with tf.data API, if possible
    metric_gen = load_data_generator(
        test_window_fns,
        test_window_timestamps,
        args.batch_size,
        shuffle=False,
        yield_batch_ts=True,
    )

    loaded_model = K.models.load_model(
        model_path
    )  # type: K.models.Model  # type: ignore

    metric_dir: Path = model_dir / "metrics"
    metric_dir.mkdir(exist_ok=True)

    if args.metric == ["all"]:
        args.metric = ["rmse", "ssim", "mse", "mae", "psnr", "corrcoef"]

    metrics: dict[str, list[float]] = {
        metric: [
            0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)
        ]  # frame-by-frame metric
        for metric in args.metric
    }
    old_count = 0
    count = 0
    first_ts: dt.datetime | None = None
    for x_olr, y_true, ts in metric_gen:  # type: ignore
        if first_ts is None:
            first_ts = ts[0]
        else:
            if first_ts in ts:
                break

        print(
            f"\rProcessing batch with timestamp: {dt.datetime.strftime(ts[-1], '%d/%m/%Y, %H:%M:%S')}",
            end="",
            flush=True,
        )
        # TODO: model_predict function with caching
        y_pred = loaded_model.predict(x_olr, verbose="0")

        old_count = count
        count += x_olr.shape[0]
        for metric in metrics.keys():
            frame_metrics = METRIC_FUNC_MAP[metric](
                y_true, y_pred
            )  # frame-by-frame metric of batch, summed up
            for i, frame_metric in enumerate(frame_metrics):
                # batch update of running mean
                metrics[metric][i] = (
                    (metrics[metric][i] * old_count) + float(frame_metric)
                ) / count

    print()
    pprint(metrics)
    with open(metric_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
