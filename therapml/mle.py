import numpy as np
from numpy.typing import NDArray


def mle_gaussian(data: NDArray, lr: float = 0.01, n_steps: int = 5000) -> dict:

    data = np.asarray(data, dtype=float)

    # Initialize mu near first data and standard deviation sigma = 1.0
    mu = float(data[0])
    sigma = 1.0

    for _ in range(n_steps):
        diff = data - mu
        mse = np.mean(diff**2)

        grad_mu = -np.mean(diff) / (sigma**2)
        grad_sigma = (1.0 / sigma) - (mse / (sigma**3))

        mu -= lr * grad_mu
        sigma -= lr * grad_sigma

    return {"mu": float(mu), "sigma2": float(sigma**2)}
