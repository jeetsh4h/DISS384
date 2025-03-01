from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt


class DataConfig:
    VAR_TYPES = ["OLR", "HEM"]

    CACHE_DIR = Path("/home/jeet/FLAME/DISS384/mosdac_cache")
    DATA_DIR = Path("/home/jeet/FLAME/DISS384/mosdac_data")

    H5_FILENAME_FMT = "3DIMG_%d%b%Y_%H%M_L2B_%s_V01R00"
    NP_FILENAME_FMT = "%Y%b%d_%H%M_3DIMG_%s"


class MOSDACConfig:
    LAT_MIN = 0
    LAT_MAX = 40
    LON_MIN = 60
    LON_MAX = 100
    LAT_LON_RESOLUTION = 0.1
    FRAME_SIZE = (
        (LAT_MAX - LAT_MIN) * int(LAT_LON_RESOLUTION**-1),
        (LON_MAX - LON_MIN) * int(LAT_LON_RESOLUTION**-1),
    )


class OLRConfig:
    MAX = 380
    MIN = 50


class HEMConfig:
    MAX = 50
    MIN = 0


class VizConfig:
    plt.rcParams["font.family"] = "monospace"

    COAST = loadmat(DataConfig.DATA_DIR / "coast.mat")
    COAST_PLOT_PARAMS = [COAST["long"], COAST["lat"], "k-"]

    OLR_IMSHOW_KWARGS = (
        {
            "extent": [
                MOSDACConfig.LON_MIN,
                MOSDACConfig.LON_MAX,
                MOSDACConfig.LAT_MIN,
                MOSDACConfig.LAT_MAX,
            ],
            "origin": "lower",
            "cmap": "Greys_r",  # TODO: change from b&w color scheme
            "vmin": OLRConfig.MIN,
            "vmax": OLRConfig.MAX,
        },
    )
    HEM_IMSHOW_KWARGS = {
        "extent": [
            MOSDACConfig.LON_MIN,
            MOSDACConfig.LON_MAX,
            MOSDACConfig.LAT_MIN,
            MOSDACConfig.LAT_MAX,
        ],
        "origin": "lower",
        "cmap": "tab20b",
        "vmin": HEMConfig.MIN,
        "vmax": HEMConfig.MAX,
    }
