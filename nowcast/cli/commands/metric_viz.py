from email.policy import default
from ...config import TFDataConfig
from ...utils.metric_viz_utils import metric_graphs


def setup_parser(subparsers):
    metric_viz_parser = subparsers.add_parser(
        "metric_viz",
        help="Visualize calculated metrics for a model.",
    )

    metric_viz_parser.add_argument(
        "--model",
        "-m",
        nargs="+",
        type=str,
        required=True,
        help="The models used to build graph. Make sure that all the models provided have different offsets. Models will only be taken from the log directory.",
    )

    metric_viz_parser.add_argument(
        "--flow",
        type=bool,
        default=False,
        help="Whether to use flow metrics, or model metrics. Model metrics by default",
    )

    return metric_viz_parser


def execute(args):
    model_dirs = [TFDataConfig.TB_LOG_DIR / model for model in args.model]
    for model_dir in model_dirs:
        if not model_dir.exists():
            print(
                f"Error: The specified model '{model_dir}' does not exist in the log directory."
            )
            return 1

        model_path = model_dir / "model.keras"
        if not model_path.exists():
            print(
                f"Error: The specified model '{model_dir}' does not contain a trained model."
            )
            return 1

    offsets = []
    for model in args.model:
        try:
            offset = int(model.split("_")[-1])
            offsets.append(offset)
        except ValueError:
            raise ValueError(
                f"Error: The model directory name '{model}' does not contain an offset."
            )

    if len(set(offsets)) != len(offsets):
        raise ValueError(
            "Error: The model directory names do not contain unique offsets."
        )

    model_metrics_dir = [
        model_dir / ("metrics" if not args.flow else "flow") for model_dir in model_dirs
    ]
    for model_metrics in model_metrics_dir:
        if not model_metrics.exists():
            print(
                f"Error: The specified model '{model_metrics}' does not contain metrics."
            )
            return 1

    # saves the figures in the log directory.
    # in the folder named `metric_viz_date_time`
    #
    metric_graphs(model_metrics_dir, offsets, args.flow)
