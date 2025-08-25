from numpy.typing import ArrayLike

import numpy as np

def projection(U3: ArrayLike) -> np.ndarray:
    # Mercator projection

    X, Y, Z = np.array(U3).T
    U2 = np.column_stack((np.arctan2(Y, X), np.log((1 + Z)/(1 - Z))/2))

    return U2
