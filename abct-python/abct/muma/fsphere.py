import numpy as np

def fsphere(p: int) -> np.ndarray:
    # Unit Fibonacci sphere
    I = np.arange(p) + 0.5
    golden_ratio = (1 + np.sqrt(5))/2
    Phi = np.acos(1 - 2*I/p)
    Theta = 2*np.pi*I / golden_ratio
    V = np.column_stack((
            np.cos(Theta)*np.sin(Phi),
            np.sin(Theta)*np.sin(Phi),
            np.cos(Phi)))

    return V
