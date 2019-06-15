
import numpy as np
from scipy.ndimage import filters
from . import masks


def susanCorner(image, radius=3.4, fptype=bool, t=25, gfrac=None,
                cgthresh=None, nshades=256):

    # Get raw corner response; many corners may consist of small clusters of
    # adjacent responsive pixels
    rawcorner = masks.usan(image, mode='Corner', radius=radius,
                           fptype=fptype, t=t, gfrac=gfrac,
                           cgthresh=cgthresh, nshades=nshades)

    # Find maximum corner response within circular USAN footprint (but force
    # footprint type to be bool in this case regardless of user-selected
    # input fptype, because float would make no sense in this context)
    fp = masks.circFootprint(radius=radius, dtype=bool)
    rawmax = filters.maximum_filter(rawcorner, footprint=fp)

    # True corners are those where response is both locally maximum as well
    # as non-zero
    corner = np.where(rawcorner == rawmax, rawcorner, 0)

    return corner
