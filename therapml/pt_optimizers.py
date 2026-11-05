from torch.optim import Optimizer
from torch import no_grad, zeros_like


class MySGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, weight_decay=0):

        super().__init__(
            params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        )

    @no_grad()
    def step(self):

        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d = p.grad

                # L2 penalty
                if wd != 0:
                    d = d + wd * p

                # Momentum buffer
                if mu != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = d.clone()
                    else:
                        state["momentum_buffer"] = mu * state["momentum_buffer"] + d
                        d = state["momentum_buffer"]

                # Parameter update
                p -= lr * d


class MyAdamW(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        super().__init__(
            params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )

    @no_grad()
    def step(self):

        for group in self.param_groups:
            lr, (b1, b2), eps, wd = (
                group["lr"],
                group["betas"],
                group["eps"],
                group["weight_decay"],
            )

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Decoupled weight decay
                p *= 1.0 - lr * wd

                # Initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = zeros_like(p)
                    state["v"] = zeros_like(p)

                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]

                # Update 1st and 2nd moments
                state["m"] = m = b1 * m + (1.0 - b1) * p.grad
                state["v"] = v = b2 * v + (1.0 - b2) * (p.grad**2)

                # Bias correction
                m_hat = m / (1.0 - b1**t)
                v_hat = v / (1.0 - b2**t)
                p -= lr * m_hat / (v_hat.sqrt() + eps)
