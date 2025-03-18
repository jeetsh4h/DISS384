import numpy as np
import datetime as dt
from pathlib import Path
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.extrapolation.semilagrangian import extrapolate

from ..config import MOSDACConfig, TFDataConfig, FlowConfig


def flow_predict(x_window, offset, ts):
    flow_dir = Path(FlowConfig.FLOW_CACHE_DIR)
    cache_fn = ts.strftime(FlowConfig.FLOW_FILENAME_FMT)
    for f in flow_dir.iterdir():
        if f.is_file() and f.name == cache_fn:
            return np.load(f)[offset : offset + TFDataConfig.HEM_WINDOW_SIZE]

    flow_field = dense_lucaskanade(
        x_window, fd_method="blob", interp_method="rbfinterp2d"
    )
    assert type(flow_field) != tuple

    # TODO: WARN: MAGIC CONSTANT
    #       Try and remove the bare constant sometime later
    #       This basically means that I am nowcasting 12 frames
    #       from the time of the nowcast.
    flow_pred = extrapolate(x_window[-1], flow_field, 12, interp_order=3)
    assert flow_pred.shape == (12, *MOSDACConfig.FRAME_SIZE)

    np.save(flow_dir / cache_fn, flow_pred)
    return flow_pred[offset : offset + TFDataConfig.HEM_WINDOW_SIZE]
