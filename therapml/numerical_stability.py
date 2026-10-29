import numpy as np
from numpy.typing import NDArray


def log_sum_exp(x: NDArray) -> float:
    x = np.asarray(x, dtype=float)
    c = np.max(x)
    return float(c + np.log(np.sum(np.exp(x - c))))
