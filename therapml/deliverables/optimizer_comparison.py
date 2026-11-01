from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wandb


def run_optimizer_with_path(opt_name, X, y, lr, n_steps, w_init, momentum=0.9):
    """Run an optimizer and record both loss history and (w1, w2) weight path."""
    n, d = X.shape
    w = w_init.copy()
    v = np.zeros(d, dtype=float)
    m = np.zeros(d, dtype=float)
    v_adam = np.zeros(d, dtype=float)

    path = [w.copy()]
    losses = []

    for t in range(1, n_steps + 1):
        diff = X @ w - y
        loss = float(np.mean(diff**2))
        losses.append(loss)
        grad = (2.0 / n) * (X.T @ diff)

        if opt_name == "SGD":
            w -= lr * grad
        elif opt_name == "Momentum":
            v = momentum * v + grad
            w -= lr * v
        elif opt_name == "Adam":
            m = 0.9 * m + 0.1 * grad
            v_adam = 0.999 * v_adam + 0.001 * (grad**2)
            m_hat = m / (1.0 - 0.9**t)
            v_hat = v_adam / (1.0 - 0.999**t)
            w -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        path.append(w.copy())

    return np.array(path), losses


def main():
    cur_dir = Path(__file__).parent

    # ill-conditioned ravine
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    X[:, 1] *= 0.2  # Make feature 2 much flatter (anisotropic curvature)
    w_true = np.array([2.0, -3.0])
    y = X @ w_true + rng.standard_normal(n) * 0.05

    # optimizers from the same initial weights
    w_init = np.array([-2.5, 3.5])
    n_steps = 150

    sgd_path, sgd_losses = run_optimizer_with_path(
        "SGD", X, y, lr=0.08, n_steps=n_steps, w_init=w_init
    )
    mom_path, mom_losses = run_optimizer_with_path(
        "Momentum", X, y, lr=0.02, n_steps=n_steps, w_init=w_init, momentum=0.85
    )
    adam_path, adam_losses = run_optimizer_with_path(
        "Adam", X, y, lr=0.15, n_steps=n_steps, w_init=w_init
    )

    # tracking
    wandb.init(
        project="therapml-training-taufik",
        name="optimizer-comparison",
        config={"n_steps": n_steps, "n_samples": n},
    )
    for step in range(n_steps):
        wandb.log(
            {
                "step": step,
                "sgd_loss": sgd_losses[step],
                "momentum_loss": mom_losses[step],
                "adam_loss": adam_losses[step],
            }
        )

    # trajectories plot
    w1_vals = np.linspace(-3.5, 3.5, 100)
    w2_vals = np.linspace(-4.5, 4.5, 100)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = np.zeros_like(W1)

    for i in range(100):
        for j in range(100):
            w_grid = np.array([W1[i, j], W2[i, j]])
            Z[i, j] = np.mean((X @ w_grid - y) ** 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contour(W1, W2, Z, levels=np.logspace(-1, 2, 20), cmap="viridis", alpha=0.6)
    ax.clabel(cp, inline=True, fontsize=8)

    ax.plot(
        sgd_path[:, 0],
        sgd_path[:, 1],
        "o-",
        color="#1f77b4",
        markersize=3,
        label="Vanilla SGD",
    )
    ax.plot(
        mom_path[:, 0],
        mom_path[:, 1],
        "s-",
        color="#ff7f0e",
        markersize=3,
        label="SGD + Momentum",
    )
    ax.plot(
        adam_path[:, 0],
        adam_path[:, 1],
        "^-",
        color="#2ca02c",
        markersize=3,
        label="Adam",
    )
    ax.scatter(
        w_true[0],
        w_true[1],
        color="red",
        marker="*",
        s=200,
        zorder=5,
        label="Optimum w*",
    )
    ax.scatter(
        w_init[0], w_init[1], color="black", marker="o", s=60, zorder=5, label="Start"
    )

    ax.set_title("2D Loss Landscape Trajectory Comparison")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.legend()
    fig.savefig(cur_dir / "optimizer_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    # Convergence Speed plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(sgd_losses, label="Vanilla SGD", color="#1f77b4")
    ax.semilogy(mom_losses, label="SGD + Momentum", color="#ff7f0e")
    ax.semilogy(adam_losses, label="Adam", color="#2ca02c")
    ax.set_title("Convergence Speed: Loss vs. Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss (Log Scale)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(cur_dir / "optimizer_convergence.png", bbox_inches="tight")
    plt.close(fig)

    # Log figures
    wandb.log(
        {
            "trajectories": wandb.Image(str(cur_dir / "optimizer_trajectories.png")),
            "convergence": wandb.Image(str(cur_dir / "optimizer_convergence.png")),
        }
    )
    wandb.finish()
    print("Logged runs and plots to Weights & Biases!")

    print(f"Saved plots to {cur_dir}/")


if __name__ == "__main__":
    main()
