from typing import Tuple
from importlib import resources

import numpy as np
from abct import muma

def mumap(*args, **kwargs) -> Tuple[np.ndarray, float]:

    # Parse, process, and test arguments
    Args = muma.step0_args(*args, **kwargs)
    Args = muma.step1_proc(Args)
    muma.step2_test(Args)

    # Initialize and run algorithm
    U = muma.step3_init(Args)
    U, CostHistory = muma.step4_run(U, Args)

    return U, CostHistory

mumap.__doc__ = resources.read_text("abct.docstrings", "mumap")
