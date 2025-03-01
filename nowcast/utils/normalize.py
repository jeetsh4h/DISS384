# TODO

OLR_MAX = 380
OLR_MIN = 50

HEM_MAX = 50
HEM_MIN = 0


def olr_normalize(olr):
    """
    Min-Max scaling for OLR values
    """
    return (olr - OLR_MIN) / (OLR_MAX - OLR_MIN)


def olr_denormalize(olr):
    """
    Denormalize OLR values
    """
    return olr * (OLR_MAX - OLR_MIN) + OLR_MIN


def hem_normalize(hem):
    """
    Min-Max scaling for HEM values
    """
    return (hem - HEM_MIN) / (HEM_MAX - HEM_MIN)


def hem_denormalize(hem):
    """
    Denormalize HEM values
    """
    return hem * (HEM_MAX - HEM_MIN) + HEM_MIN
