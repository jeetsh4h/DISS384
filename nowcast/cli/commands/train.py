import json
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

from ...utils.model_utils import *
from ...utils.models import encoder_decoder
from ...utils.metric_viz_utils import training_graphs
from ...config import TrainConfig, MOSDACConfig, TFDataConfig
from ...utils.train_utils import model_callbacks, generate_log_dir
from ...utils.data_loader import load_data_generator, create_windows


def setup_parser(subparsers):
    train_parser = subparsers.add_parser(
        "train",
        help=(
            "Train a model to nowcast OLR to HEM. "
            + "Currently, it only pulls data from the cache directory. "
            + "Run the `cache` command first, and then `train`"
        ),
    )

    train_parser.add_argument(
        "--offset",
        "-no",
        type=int,
        required=True,
        help="The offset at which HEM is nowcasted from the input OLR",
    )

    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=24,
        help="Number of epochs to train the model",
    )

    train_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Batch size for training",
    )

    train_parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        default=TFDataConfig.TB_LOG_DIR,
        help="Directory to save TensorBoard logs. To view the board, run `tensorboard --logdir <logdir>`",
    )

    train_parser.add_argument(
        "--train_metric_viz",
        "-tm",
        type=str,
        help="This can be run for a model after the training has occurred. Currently taking the model log directory as input, and outputting the loss graph across the epochs (for training and validation data).\nRemember to add offset, it will glean the offset from the logdir name, it will throw an error if not provided, or it does not match. Can NOT train a model and visualize metrics, have to train model first.",
    )

    return train_parser


def execute(args):
    epochs = args.epochs
    batch_size = args.batch_size
    offset = args.offset

    if args.train_metric_viz:
        model_train_dir = TFDataConfig.TB_LOG_DIR / args.train_metric_viz
        if not model_train_dir.exists():
            print(
                f"Error: The specified model '{args.train_metric_viz}' does not exist in the log directory."
            )
            return 1

        assert offset == int(
            args.train_metric_viz.split("_")[-1]
        ), "Offset does not match the model directory name."

        model_train_metric_path = model_train_dir / "training_history.json"
        if not model_train_metric_path.exists():
            print(
                f"Error: The specified model '{args.train_metric_viz}' does not contain training metrics."
            )
            return 1

        training_history = None
        with open(model_train_metric_path, "r") as f:
            training_history = json.load(f)

        graph = training_graphs(training_history, offset)
        graph.savefig(model_train_dir / "loss_metrics.png")
        plt.close(graph)

        return 0

    train_window_fns, train_window_timestamps = create_windows(
        TrainConfig.TRAIN_START_DT,
        TrainConfig.TRAIN_END_DT,
        offset,
    )
    train_data_gen = tf.data.Dataset.from_generator(
        generator=lambda: load_data_generator(
            train_window_fns,
            train_window_timestamps,
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

    val_window_fns, val_window_timestamps = create_windows(
        TrainConfig.VAL_START_DT,
        TrainConfig.VAL_END_DT,
        offset,
    )
    val_data_gen = tf.data.Dataset.from_generator(
        generator=lambda: load_data_generator(
            val_window_fns,
            val_window_timestamps,
            batch_size,
            shuffle=True,
            yield_batch_ts=False,
        ),
        output_signature=(
            tf.TensorSpec(
                shape=(None, TFDataConfig.HEM_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1),  # type: ignore
                dtype=TFDataConfig.DTYPE,  # type: ignore
            ),
            tf.TensorSpec(
                shape=(None, TFDataConfig.OLR_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1),  # type: ignore
                dtype=TFDataConfig.DTYPE,  # type: ignore
            ),
        ),
    )

    model = encoder_decoder()

    # Compile the model with appropriate metrics
    model.compile(
        optimizer="adam",
        loss=CombinedLoss(use_focal=True),
        # loss=weighted_pixel_loss,
        # loss=weighted_denorm_rmse,
        # loss=combined_loss,
        # loss=frame_loss,
        metrics=[
            denorm_rmse,
            non_zero_denorm_rmse,
            weighted_denorm_rmse,
            FocalLoss(),
            # frame_loss,
            # weighted_pixel_loss,
        ],
    )

    # Log directory for Model
    logdir = generate_log_dir(Path(args.logdir), args.offset)

    # Train the model with callbacks
    history = model.fit(
        train_data_gen.prefetch(tf.data.experimental.AUTOTUNE),
        steps_per_epoch=(
            len(train_window_fns) // batch_size
            if len(train_window_fns) % batch_size == 0
            else (len(train_window_fns) // batch_size) + 1
        ),
        validation_data=val_data_gen.prefetch(tf.data.experimental.AUTOTUNE),
        validation_steps=(
            len(val_window_fns) // batch_size
            if len(val_window_fns) % batch_size == 0
            else (len(val_window_fns) // batch_size) + 1
        ),
        epochs=epochs,
        callbacks=model_callbacks(logdir),
        verbose="auto",
    )

    # Save model
    model.save(logdir / "model.keras")

    # Save training history to a JSON file
    history_dict = history.history
    with open(logdir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=4)

    print(f"Model and training history saved to {logdir}")
    print(f"To view TensorBoard, run: `tensorboard --logdir {logdir}`")

    graph = training_graphs(history_dict, offset)
    graph.savefig(logdir / "loss_metrics.png")
    plt.close(graph)
