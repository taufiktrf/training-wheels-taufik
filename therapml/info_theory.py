import numpy as np
from numpy.typing import NDArray


def entropy(P: NDArray) -> float:
    P = np.asarray(P, dtype=float)
    mask = P > 0
    return float(-np.sum(P[mask] * np.log2(P[mask])))


def kl_divergence(P: NDArray, Q: NDArray) -> float:
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    mask = P > 0
    return float(np.sum(P[mask] * np.log2(P[mask] / Q[mask])))


def cross_entropy(P: NDArray, Q: NDArray) -> float:
    return entropy(P) + kl_divergence(P, Q)


def mutual_information(joint: NDArray) -> float:
    joint = np.asarray(joint, dtype=float)
    px = np.sum(joint, axis=1, keepdims=True)
    py = np.sum(joint, axis=0, keepdims=True)
    independent = px @ py

    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / independent[mask])))
