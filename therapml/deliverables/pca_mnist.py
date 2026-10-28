from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from therapml.pca import pca


def main():
    cur_dir = Path(__file__).parent
    sample_idx = 578

    ds = MNIST(root=".mnist_data", train=False, download=True, transform=ToTensor())
    X = ds.data.numpy().reshape(-1, 784).astype(float) / 255.0
    y = ds.targets.numpy()

    # 2D projection
    Z = pca(X, k=2)["projected"]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", alpha=0.5, s=3)
    plt.colorbar(scatter, label="Digit")
    plt.title("MNIST 2D PCA Projection")
    plt.savefig(cur_dir / "pca_mnist_2d_projection.png", bbox_inches="tight")
    plt.close()

    # Reconstruction error & sample reconstruction
    ks = [2, 10, 50, 100, 784]
    errors, recons = [], []
    for k in ks:
        r = pca(X, k=k)["reconstructed"]
        recons.append(r[sample_idx].reshape(28, 28))
        errors.append(np.linalg.norm(X - r, "fro"))
        print(f"k={k:<3} | Error: {errors[-1]:.2f}")

    plt.figure(figsize=(7, 4))
    plt.plot(ks, errors, "o-")
    plt.title("Reconstruction Error vs k")
    plt.xlabel("k")
    plt.ylabel("Frobenius Error")
    plt.savefig(cur_dir / "pca_mnist_reconstruction_error.png", bbox_inches="tight")
    plt.close()

    # Visual comparison of a sample digit
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(11, 2.5))
    axes[0].imshow(X[sample_idx].reshape(28, 28), cmap="gray")
    axes[0].set_title(f"Orig ({y[sample_idx]})")
    axes[0].axis("off")
    for ax, k, img in zip(axes[1:], ks, recons):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"k={k}")
        ax.axis("off")
    plt.savefig(cur_dir / "pca_mnist_sample_reconstruction.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
