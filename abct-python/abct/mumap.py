from typing import Tuple
from importlib import resources

import numpy as np
from abct import loyv

def mumap(*args, **kwargs) -> Tuple[np.ndarray, float]:

    # Parse, process, and test arguments
    Args = mumap.step0_args(*args, **kwargs)
    Args = mumap.step1_proc(Args)
    mumap.step2_test(Args)

    # Initialize and run algorithm
    U = mumap.step3_init(Args)
    U, CostHistory = mumap.step4_run(U, Args)

    return U, CostHistory

mumap.__doc__ = resources.read_text("abct.docstrings", "mumap")
