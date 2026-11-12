import numpy as np

def stationary_gth(T):
    """Stationary distribution of a stochastic matrix T (rows sum to one) by
    Grassmann-Taksar-Heyman elimination (no subtraction)."""
    P = np.array(T, dtype=float)
    n = len(P)
    for k in range(n - 1, 0, -1):
        s = P[k, :k].sum()
        P[:k, k] /= s
        P[:k, :k] += np.outer(P[:k, k], P[k, :k])
    x = np.zeros(n); x[0] = 1.0
    for k in range(1, n):
        x[k] = x[:k] @ P[:k, k]
    return x / x.sum()

def coop_sml(rho, coop, S):
    """Long-run cooperation in the small-mutation limit for the strategy subset S.
    rho[a, b]: fixation probability of one a-mutant in a b-population; coop[s]:
    cooperation rate of a homogeneous s-population."""
    S = list(S); K = len(S)
    T = np.zeros((K, K))
    for i, b in enumerate(S):            # from resident b ...
        for j, a in enumerate(S):        # ... to mutant a
            if i != j:
                T[i, j] = rho[a, b] / (K - 1)
        T[i, i] = 1.0 - T[i].sum()
    return stationary_gth(T) @ coop[S]

def ssfi_level(rho, coop, SA, U, theta=0.5, draws=20000, k=None, seed=1):
    """Superset SSFI of the level claim C(S) > theta (k=None: each omitted
    strategy included with probability 1/2; integer k: exactly k added)."""
    rng = np.random.default_rng(seed)
    O = [s for s in U if s not in SA]
    vA = coop_sml(rho, coop, SA) > theta
    flips = 0
    for _ in range(draws):
        R = [s for s in O if rng.random() < 0.5] if k is None else list(rng.choice(O, k, replace=False))
        flips += (coop_sml(rho, coop, list(SA) + R) > theta) != vA
    return flips / draws
