import numpy as np
from numpy.typing import NDArray


def truncated_svd(matrix: NDArray, rank: int) -> dict:

    # Correct shape handling
    m, n = matrix.shape
    A = matrix.T if m < n else matrix

    # Eigendecomposition and sorting
    C = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1][:rank]
    S = np.sqrt(np.maximum(eigenvalues[idx], 0.0))
    V = eigenvectors[:, idx]

    # Compute singular vectors
    safe_S = np.where(S > 1e-12, S, 1.0)
    U = np.where(S > 1e-12, (A @ V) / safe_S, 0.0)
    Vt = V.T

    # if original matrix had m < n, swap U and V
    if m < n:
        U, Vt = Vt.T, U.T

    # reconstruction
    reconstructed = U @ np.diag(S) @ Vt

    return {
        "U": U,
        "S": S,
        "Vt": Vt,
        "reconstructed": reconstructed,
    }
