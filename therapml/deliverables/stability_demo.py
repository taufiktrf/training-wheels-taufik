import numpy as np
from therapml.numerical_stability import log_sum_exp


def demo_underflow():
    print("1. Underflow: Multiplying Small Probabilities")
    p = 0.01
    prob = 1.0
    steps = 0

    # Multiply until 0.0
    while prob > 0:
        prob *= p
        steps += 1

    print(f"Multiplying p = {p}:")
    print(f"  Naive product after {steps} steps = {prob} (Underflow to 0.0)")
    print(f"  Naive sum of two paths (P + P)    = {prob + prob} -> log(0) = {-np.inf}")

    # Fix: accumulate in log-space
    log_p = steps * np.log(p)
    stable_log_sum = log_sum_exp(np.array([log_p, log_p]))

    print(f"  Fix via Log-Sum-Exp: log(P + P)   = {stable_log_sum:.4f}\n")


def demo_overflow():
    print("2. Overflow: Large Exponents")
    x = np.array([1000.0, 1001.0, 1002.0])

    # overflows to inf
    with np.errstate(over="ignore"):
        naive = np.log(np.sum(np.exp(x)))

    print(f"Inputs: {x}")
    print(f"  Naive log(sum(exp(x))) = {naive} (Overflow to inf)")

    # Fix: stable Log-Sum-Exp
    stable = log_sum_exp(x)
    print(f"  Fix via Log-Sum-Exp    = {stable:.4f}\n")


if __name__ == "__main__":
    demo_underflow()
    demo_overflow()
