import numpy as np
from numpy.typing import NDArray


def sgd_fit(X: NDArray, y: NDArray, lr: float, n_steps: int) -> dict:
    """Vanilla Gradient Descent on Mean Squared Error (MSE)."""
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    losses = []

    for _ in range(n_steps):
        diff = X @ w - y
        losses.append(float(np.mean(diff**2)))
        grad = (2.0 / n) * (X.T @ diff)
        w -= lr * grad

    return {"weights": w, "loss_history": losses}


def sgd_momentum_fit(
    X: NDArray, y: NDArray, lr: float, momentum: float, n_steps: int
) -> dict:
    """SGD with momentum velocity accumulation: v = beta * v + g."""
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    v = np.zeros(d, dtype=float)
    losses = []

    for _ in range(n_steps):
        diff = X @ w - y
        losses.append(float(np.mean(diff**2)))
        grad = (2.0 / n) * (X.T @ diff)
        v = momentum * v + grad
        w -= lr * v

    return {"weights": w, "loss_history": losses}


def adam_fit(
    X: NDArray,
    y: NDArray,
    lr: float,
    n_steps: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> dict:
    """Adam optimizer with 1st and 2nd moment bias corrections."""
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    m = np.zeros(d, dtype=float)
    v = np.zeros(d, dtype=float)
    losses = []

    for t in range(1, n_steps + 1):
        diff = X @ w - y
        losses.append(float(np.mean(diff**2)))
        grad = (2.0 / n) * (X.T @ diff)

        # Update biased 1st and 2nd moment estimates
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)

        # Bias corrections
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        w -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return {"weights": w, "loss_history": losses}
