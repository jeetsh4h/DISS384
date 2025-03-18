import json
import datetime as dt
from pathlib import Path
from pprint import pprint


from ...utils.metric_utils import *
from ...utils.flow_utils import flow_predict
from ...utils.file_utils import find_by_date
from ...utils.normalize import _fill_nans_with_interpolation
from ...config import DataConfig, FlowConfig, TFDataConfig, TestConfig


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

    dates = []
    curr_date = TestConfig.TEST_START_DT
    assert curr_date.minute == 30 or curr_date.minute == 0
    while curr_date <= TestConfig.TEST_END_DT:
        dates.append(curr_date)
        curr_date += dt.timedelta(minutes=30)

    for date in dates:
        print(f"\rProcessing {date.strftime('%Y-%m-%d %H:%M')}...", end="")
        try:
            hem_fn, hem_ts = find_by_date(
                date,
                date
                + dt.timedelta(minutes=30 * (TFDataConfig.HEM_WINDOW_SIZE))
                + dt.timedelta(
                    minutes=30 * (offset + TFDataConfig.HEM_WINDOW_SIZE - 1)
                ),
                str(DataConfig.CACHE_DIR / "HEM"),
                str(DataConfig.NP_FILENAME_FMT).replace("%s", "HEM", 1),
                "npy",
                30,
                0,
                False,
            )

        except OSError:
            continue

        inp_window_fn = hem_fn[: TFDataConfig.HEM_WINDOW_SIZE]
        out_window_fn = hem_fn[TFDataConfig.HEM_WINDOW_SIZE + offset :]

        if None in inp_window_fn or None in out_window_fn:
            continue

        inp_window = [
            np.clip(
                (_fill_nans_with_interpolation(np.load(fn))),  # type: ignore
                HEMConfig.MIN,
                HEMConfig.MAX,
            )
            for fn in inp_window_fn
        ]
        out_window = [
            np.clip(
                (_fill_nans_with_interpolation(np.load(fn))),  # type: ignore
                HEMConfig.MIN,
                HEMConfig.MAX,
            )
            for fn in out_window_fn
        ]

        x_hem_flow = np.array(inp_window)
        y_true_flow = np.array(out_window)

        y_pred = flow_predict(x_hem_flow, offset, date)

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
