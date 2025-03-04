import numpy as np
import keras.api as K
import datetime as dt
from pathlib import Path
import matplotlib.pyplot as plt


from ...config import MOSDACConfig, TrainConfig
from ...utils.viz_utils import visualize_hem_compare, window_by_date
from ...utils.normalize import hem_denormalize


def setup_parser(subparsers):
    visualize_parser = subparsers.add_parser(
        "visualize",
        help=(
            "Visualizs model's prediction v/s truth (only windows). "
            + "Currently, it only pulls data from the cache directory for the truth. "
            + "Run the `cache` command first, and then `visualize`. "
            + "The model is loaded from one of folders from the log directory."
        ),
    )

    visualize_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="The model to visualize, from the log directory only (for now).",
    )

    visualize_parser.add_argument(
        "--dates",
        "-dt",
        nargs="+",
        help="Dates in format 'YYYY-MM-DD_HH:MM'",
    )

    visualize_parser.add_argument(
        "--date-ranges",
        "-d",
        nargs="+",
        help="Date ranges in format 'YYYY-MM-DD:YYYY-MM-DD' (assumed to be 00:00 for start and 23:59 for end) or 'YYYY-MM-DD_HH:MM:YYYY-MM-DD_HH:MM'",
    )

    visualize_parser.add_argument(
        "--offset",
        "-no",
        type=int,
        help="The offset at which HEM is nowcasted from the input OLR. Will be taken from the model's directory name if not provided.",
    )

    visualize_parser.add_argument(
        "--log-norm",
        "-ln",
        type=bool,
        default=False,
        help="Log the normalized prediction as an image.",
    )

    return visualize_parser


def execute(args):
    model_dir: Path = TrainConfig.TensorBoardLogDir / args.model
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

    dates: list[dt.datetime] = []
    if (args.dates is not None) and (args.date_ranges is not None):
        raise ValueError(
            "Error: Both dates and date ranges cannot be provided at the same time."
        )

    if args.dates is not None:
        for date_str in args.dates:
            try:
                parsed_date = dt.datetime.strptime(date_str, "%Y-%m-%d_%H:%M")
            except:
                raise ValueError(
                    f"Error: Invalid date format '{date_str}'. Use 'YYYY-MM-DD_HH:MM'."
                )
            dates.append(_round_to_next_30_minute_mark(parsed_date))

    if args.date_ranges is not None:
        for date_range in args.date_ranges:
            try:
                start_date, end_date = parse_date_range(date_range)
            except ValueError as e:
                raise ValueError(f"Error: {str(e)}")

            # Use the function
            current_date = _round_to_next_30_minute_mark(start_date)

            while current_date <= end_date:
                dates.append(current_date)
                current_date += dt.timedelta(minutes=30)

    assert dates and (offset is not None)

    loaded_model = K.models.load_model(
        model_path
    )  # type: K.models.Model  # type: ignore

    fig_dir = model_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    for date in dates:
        olr_norm_data, hem_norm_data = window_by_date(date, offset)
        if olr_norm_data is None or hem_norm_data is None:
            continue

        assert (olr_norm_data is not None) and (hem_norm_data is not None)

        hem_norm_prediction = loaded_model.predict(olr_norm_data[np.newaxis, ...], batch_size=1)  # type: ignore

        # log normalized prediction as an image
        if args.log_norm:
            fig, axs = plt.subplots(1, 4, figsize=(16, 8))
            for i, ax in enumerate(axs):
                ax.imshow(
                    hem_norm_prediction[0, ..., 0][i],
                    origin="lower",
                    extent=[
                        MOSDACConfig.LON_MIN,
                        MOSDACConfig.LON_MAX,
                        MOSDACConfig.LAT_MIN,
                        MOSDACConfig.LAT_MAX,
                    ],
                    cmap="tab20b",
                    vmin=0,
                    vmax=1,
                )
                ax.set_title(f"Frame {i + 1}")

            fig.savefig("./log.png")
            plt.close(fig)

        hem_denorm_prediction = hem_denormalize(hem_norm_prediction[0, ..., 0])

        hem_denorm_truth = hem_denormalize(hem_norm_data[..., 0])

        assert (
            hem_denorm_prediction.shape == hem_denorm_truth.shape
            and hem_denorm_prediction.shape
            == (TrainConfig.HEM_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE)
        )

        fig = visualize_hem_compare(
            hem_denorm_prediction, hem_denorm_truth, date, offset
        )

        fig.savefig(fig_dir / f"{date.strftime('%Y-%m-%d_%H:%M')}.png")
        plt.close(fig)


def parse_date(date_str: str, start_date: bool) -> dt.datetime:
    """Parse date string in YYYY-MM-DD or YYYY-MM-DD_HH:MM format."""
    date_parsing_fmt = "%Y-%m-%d_%H:%M"
    try:
        if "_" in date_str:
            return dt.datetime.strptime(date_str, date_parsing_fmt)
        else:
            return (
                dt.datetime.strptime(date_str + "_00:00", date_parsing_fmt)
                if start_date
                else dt.datetime.strptime(date_str + "_23:59", date_parsing_fmt)
            )
    except ValueError:
        raise ValueError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD_HH:MM"
        )


def parse_date_range(date_range: str) -> tuple[dt.datetime, dt.datetime]:
    """Parse date range string in the format 'start_date:end_date'."""
    try:
        start_date_str, end_date_str = date_range.split(":")
        start_date = parse_date(start_date_str, True)
        end_date = parse_date(end_date_str, False)

        if start_date >= end_date:
            raise ValueError("Start date must be earlier than end date")

        if start_date.year != end_date.year:
            raise ValueError("Start and end date must be in the same year")

        return start_date, end_date
    except ValueError as e:
        raise ValueError(f"Invalid date range format: {date_range}. {str(e)}")


def _round_to_next_30_minute_mark(date: dt.datetime) -> dt.datetime:
    """
    Rounds a datetime to the next 30-minute mark.
    If already on a 30-minute mark, returns the same datetime.
    """
    minutes_since_hour = date.minute
    if minutes_since_hour % 30 == 0:
        # Already on a 30-minute mark
        return date
    else:
        # Round up to next 30-minute mark
        minutes_to_add = 30 - (minutes_since_hour % 30)
        return date + dt.timedelta(minutes=minutes_to_add)
