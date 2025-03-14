import json
import keras.api as K
import tensorflow as tf
from pathlib import Path

from ...config import MOSDACConfig, TestConfig, TFDataConfig
from ...utils.data_loader import create_windows, load_data_generator


def setup_parser(subparsers):
    test_parser = subparsers.add_parser(
        "test",
        help=(
            "Test a trained model on a set of test data. "
            + "Currently, it only pulls data from the cache directory. "
            + "Run the `cache` command first, and then `test`."
            + "Tests the model on all the data outlined as per the config."
        ),
    )

    test_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="The model to test, from the log directory only (for now).",
    )

    test_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for evaluating",
    )

    test_parser.add_argument(
        "--offset",
        "-no",
        type=int,
        help="The offset at which HEM is nowcasted from the input OLR. Will be taken from the model's directory name if not provided.",
    )

    return test_parser


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
    batch_size = args.batch_size

    test_window_fns, test_window_timestamps = create_windows(
        TestConfig.TEST_START_DT,
        TestConfig.TEST_END_DT,
        offset,
    )
    test_data_gen = tf.data.Dataset.from_generator(
        generator=lambda: load_data_generator(
            test_window_fns,
            test_window_timestamps,
            batch_size,
            shuffle=True,
            yield_batch_ts=False,
        ),
        output_signature=(
            tf.TensorSpec(
                shape=(None, TFDataConfig.OLR_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1),  # type: ignore
                dtype=TFDataConfig.DTYPE,  # type: ignore
            ),
            tf.TensorSpec(
                shape=(None, TFDataConfig.HEM_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1),  # type: ignore
                dtype=TFDataConfig.DTYPE,  # type: ignore
            ),
        ),
    )

    loaded_model = K.models.load_model(
        model_path
    )  # type: K.models.Model  # type: ignore

    test_dir = model_dir / "test"
    test_dir.mkdir(exist_ok=True)

    # Evaluate the model
    test_metrics = loaded_model.evaluate(
        test_data_gen.prefetch(tf.data.experimental.AUTOTUNE),
        steps=(
            len(test_window_fns) // batch_size
            if len(test_window_fns) % batch_size == 0
            else len(test_window_fns) // batch_size + 1
        ),
        verbose="auto",
        return_dict=True,
    )
    with open(test_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
