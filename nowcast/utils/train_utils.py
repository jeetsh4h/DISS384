import datetime as dt
import keras.api as K
from pathlib import Path


def model_callbacks(logdir: Path) -> list[K.callbacks.Callback]:
    return [
        K.callbacks.TensorBoard(
            log_dir=str(logdir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        ),
        K.callbacks.ModelCheckpoint(
            filepath=str(logdir / "model_checkpoint.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        K.callbacks.BackupAndRestore(backup_dir=str(logdir / "backup")),
        K.callbacks.TerminateOnNaN(),
        K.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
    ]


def generate_log_dir(base_logdir: Path, nowcast_offset: int) -> Path:
    """Generate a timestamped log directory for TensorBoard"""
    current_time = dt.datetime.now(
        dt.timezone(dt.timedelta(hours=5, minutes=30))
    ).strftime(f"%Y%m%d-%H%M%S_offset_{nowcast_offset}")

    return base_logdir / current_time
