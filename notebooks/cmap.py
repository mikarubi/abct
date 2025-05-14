from typing import List
from numpy.typing import ArrayLike

import numpy as np

def get_cmap(cmap: ArrayLike) -> List[str]:
    """
    Convert a (n, 3) numpy array of normalized colors to a list of (i/n, rgb(r, g, b)) tuples.
    """

    cmap = np.asarray(cmap)

    n = len(cmap)
    rgb = [None] * n
    for i, c in enumerate(cmap):
        r, g, b = (255 * c).round().astype(int)
        rgb[i] = f"rgb({r}, {g}, {b})"
        
    return rgb
