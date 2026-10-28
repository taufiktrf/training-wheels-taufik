from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from therapml.svd import truncated_svd


def main():
    cur_dir = Path(__file__).parent

    img = plt.imread(cur_dir / "grayscale.jpg").astype(float)
    if img.ndim == 3:
        img = img @ [0.2989, 0.5870, 0.1140]

    ranks = [1, 5, 10, 50, 100]
    errors, recons = [], []
    for r in ranks:
        recon = truncated_svd(img, rank=r)["reconstructed"]
        recons.append(recon)
        errors.append(np.linalg.norm(img - recon, "fro"))
        print(f"r={r:<3} | Error: {errors[-1]:.2f}")

    # Error curve
    plt.figure(figsize=(7, 4))
    plt.plot(ranks, errors, "o-g")
    plt.title("SVD Reconstruction Error vs Rank")
    plt.xlabel("Rank (r)")
    plt.ylabel("Frobenius Error")
    plt.savefig(cur_dir / "svd_reconstruction_error.png", bbox_inches="tight")
    plt.close()

    # Visual comparison
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(12, 2.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    for ax, r, rec in zip(axes[1:], ranks, recons):
        ax.imshow(rec, cmap="gray")
        ax.set_title(f"r={r}")
        ax.axis("off")
    plt.savefig(cur_dir / "svd_image_reconstructions.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
