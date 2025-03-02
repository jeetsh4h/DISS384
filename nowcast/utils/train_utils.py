import datetime as dt
import keras.api as K


def model_callbacks(args) -> list[K.callbacks.Callback]:
    log_dir = _generate_log_dir(args.base_logdir, args.offset)

    return [
        K.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        ),
        K.callbacks.ModelCheckpoint(
            filepath=log_dir / "model_checkpoint.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        K.callbacks.BackupAndRestore(backup_dir=log_dir / "backup"),
        K.callbacks.TerminateOnNaN(),
        K.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
    ]


def _generate_log_dir(base_logdir, nowcast_offset):
    """Generate a timestamped log directory for TensorBoard"""
    current_time = dt.datetime.now(
        dt.timezone(dt.timedelta(hours=5, minutes=30))
    ).strftime(f"%Y%m%d-%H%M%S_offset_{nowcast_offset}")
    return base_logdir / current_time
