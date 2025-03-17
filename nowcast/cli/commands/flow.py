import json
import datetime as dt
from pathlib import Path
from pprint import pprint

from nowcast.utils.normalize import hem_denormalize

from ...utils.metric_utils import *
from ...utils.flow_utils import flow_predict
from ...config import FlowConfig, TFDataConfig, TestConfig
from ...utils.data_loader import create_windows, load_data_generator


def setup_parser(subparsers):
    flow_parser = subparsers.add_parser(
        "flow",
        help="Nowcast HEM using optical flow. Do nowcasting that directly corresponds to what one model is doing. Calulating baseline metrics for a model.",
    )

    flow_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="The model to compare baseline with, from the log directory only (for now).",
    )

    flow_parser.add_argument(
        "--metric",
        "-me",
        nargs="+",
        type=str,
        choices=["rmse", "corrcoef", "ssim", "mse", "mae", "psnr", "all"],
        required=True,
        help="One or more metric names to compute.",
    )

    flow_parser.add_argument(
        "--offset",
        "-no",
        type=int,
        help="The offset at which HEM is nowcasted from the input OLR. Will be taken from the model's directory name if not provided.",
    )

    return flow_parser


def execute(args):
    # check if flow (HEM) cache folder exists or not
    FlowConfig.FLOW_CACHE_DIR.mkdir(exist_ok=True)

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
        TestConfig.TEST_START_DT, TestConfig.TEST_END_DT, offset, hem_as_input=True
    )
    flow_gen = load_data_generator(
        test_window_fns,
        test_window_timestamps,
        batch_size=1,
        shuffle=False,
        yield_batch_ts=True,
    )

    if args.metric == ["all"]:
        # args.metric = ["rmse", "corrcoef", "ssim", "mse", "mae", "psnr"]
        args.metric = ["mse", "rmse", "mae", "psnr"]

    flow_dir: Path = model_dir / "flow"
    flow_dir.mkdir(exist_ok=True)

    metrics: dict[str, list[float]] = {
        metric: [
            0.0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)
        ]  # frame-by-frame metric
        for metric in args.metric
    }
    count = {
        metric: [0 for _ in range(TFDataConfig.HEM_WINDOW_SIZE)]
        for metric in args.metric
    }
    first_ts: dt.datetime | None = None
    for x_hem, y_true, ts in flow_gen:  # type: ignore
        if first_ts is None:
            first_ts = ts[0]
        else:
            if first_ts in ts:
                break

        print(
            f"\rProcessing window with timestamp: {dt.datetime.strftime(ts[-1], '%d/%m/%Y, %H:%M:%S')}",
            end="",
            flush=True,
        )
        x_hem_flow = hem_denormalize(x_hem[0, ..., 0])
        y_true_flow = hem_denormalize(y_true[0, ..., 0])

        y_pred = flow_predict(x_hem_flow, offset, ts[0])

        for metric in metrics.keys():
            frame_metrics = FLOW_METRIC_FUNC_MAP[metric](
                y_true_flow, y_pred
            )  # frame-by-frame metric of batch

            for i, frame_metric in enumerate(frame_metrics):
                if frame_metric is not None:
                    count[metric][i] += 1
                    metric_frame_count = count[metric][i]
                    metrics[metric][i] = (
                        metrics[metric][i] * (metric_frame_count - 1)
                        + float(frame_metric)
                    ) / metric_frame_count

    print()
    pprint(metrics)
    with open(flow_dir / "flow_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
