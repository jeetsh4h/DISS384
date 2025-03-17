import numpy as np
import datetime as dt
from pathlib import Path
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.extrapolation.semilagrangian import extrapolate

from ..config import TFDataConfig, FlowConfig


# TODO: check the kwaargs,,, there are better ones somewhere
def flow_predict(x_window, offset, ts: dt.datetime) -> np.ndarray:
    flow_dir = Path(FlowConfig.FLOW_CACHE_DIR)
    filename = ts.strftime(FlowConfig.FLOW_FILENAME_FMT)
    flow_files = [f.name for f in flow_dir.iterdir() if f.is_file()]

    if filename not in flow_files:
        flow_field = dense_lucaskanade(x_window)

        assert type(flow_field) != tuple

        # TODO: WARN: MAGIC CONSTANT
        #       Try and remove the bare constant sometime later
        #       This basically means that I am nowcasting 12 frames
        #       from the time of the nowcast.
        y_pred = extrapolate(x_window[-1], flow_field, 12)
        np.save(flow_dir / filename, y_pred)
        return y_pred[offset : offset + TFDataConfig.HEM_WINDOW_SIZE]

    else:
        flow_nowcasts = np.load(flow_dir / filename)
        return flow_nowcasts[offset : offset + TFDataConfig.HEM_WINDOW_SIZE]
