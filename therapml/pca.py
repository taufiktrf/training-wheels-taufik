import numpy as np
from numpy.typing import NDArray


def pca(data: NDArray, k: int) -> dict:
    # Mean centering
    mean = np.mean(data, axis=0)
    centered = data - mean

    # Covariance matrix (features as columns)
    cov = np.cov(centered, rowvar=False)

    # Eigendecomposition (eigh for symmetric matrix) and then components
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    top_idx = idx[:k]
    top_eigenvalues = eigenvalues[top_idx]
    components = eigenvectors[:, top_idx].T

    # Projection & Reconstruction
    projected = centered @ components.T
    reconstructed = projected @ components + mean

    return {
        "projected": projected,
        "components": components,
        "eigenvalues": top_eigenvalues,
        "mean": mean,
        "reconstructed": reconstructed,
    }
