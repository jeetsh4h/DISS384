import tensorflow as tf

from ...utils.models import encoder_decoder
from ...config import TrainConfig, MOSDACConfig
from ...utils.train_utils import model_callbacks
from ...utils.data_loader import load_data_generator, create_windows
from ...utils.model_utils import denorm_rmse, weighted_denorm_rmse, non_zero_denorm_rmse


def setup_parser(subparsers):
    train_parser = subparsers.add_parser(
        "train", help="Train a model to nowcast OLR to HEM."
    )

    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=16,
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
        "--offset",
        "-no",
        type=int,
        required=True,
        help="The offset at which HEM is nowcasted from the input OLR",
    )

    train_parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        default=TrainConfig.TensorBoardLogDir,
        help="Directory to save TensorBoard logs",
    )


def execute(args):
    epochs = args.epochs
    batch_size = args.batch_size

    train_window_fns, train_window_timestamps = create_windows(
        TrainConfig.TRAIN_START_DT,
        TrainConfig.TRAIN_END_DT,
        TrainConfig.HEM_WINDOW_SIZE,
    )
    train_data_gen = tf.data.Dataset.from_generator(
        load_data_generator(
            train_window_fns,
            train_window_timestamps,
            batch_size,
            shuffle=True,
            yield_batch_ts=False,
        ),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, TrainConfig.OLR_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1), dtype=tf.float16),  # type: ignore
            tf.TensorSpec(shape=(batch_size, TrainConfig.HEM_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1), dtype=tf.float16),  # type: ignore
        ),
    )

    val_window_fns, val_window_timestamps = create_windows(
        TrainConfig.VAL_START_DT,
        TrainConfig.VAL_END_DT,
        TrainConfig.HEM_WINDOW_SIZE,
    )
    val_data_gen = tf.data.Dataset.from_generator(
        load_data_generator(
            val_window_fns,
            val_window_timestamps,
            batch_size,
            shuffle=False,
            yield_batch_ts=False,
        ),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, TrainConfig.OLR_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1), dtype=tf.float16),  # type: ignore
            tf.TensorSpec(shape=(batch_size, TrainConfig.HEM_WINDOW_SIZE, *MOSDACConfig.FRAME_SIZE, 1), dtype=tf.float16),  # type: ignore
        ),
    )

    model = encoder_decoder()

    # Compile the model with appropriate metrics
    model.compile(
        optimizer="adam",
        loss=weighted_denorm_rmse,
        metrics=[non_zero_denorm_rmse, denorm_rmse],
    )

    # Train the model with callbacks
    history = model.fit(
        train_data_gen.prefetch(tf.data.experimental.AUTOTUNE),
        steps_per_epoch=(
            len(train_window_fns) // batch_size
            if len(train_window_fns) % batch_size == 0
            else len(train_window_fns) // batch_size + 1
        ),
        validation_data=val_data_gen.prefetch(tf.data.experimental.AUTOTUNE),
        validation_steps=(
            len(val_window_fns) // batch_size
            if len(val_window_fns) % batch_size == 0
            else len(val_window_fns) // batch_size + 1
        ),
        epochs=epochs,
        callbacks=model_callbacks(args),
        verbose="auto",
    )
