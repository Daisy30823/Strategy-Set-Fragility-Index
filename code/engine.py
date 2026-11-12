"""Core engine for the strategy-set multiverse audit.

Three model families are implemented with exact (non-simulated) calculations:
  * DR  repeated prisoner's dilemma, memory-one strategies, discounted payoffs,
  * PU  optional public goods game with peer punishment (group sampling),
  * IR  indirect reciprocity with public binary reputations and second-order norms.

For every family the engine returns, for a universe U of strategies, the
ordered-pair fixation-probability matrix rho[i, j] = probability that a single
mutant of strategy i takes over a resident population of strategy j, and a
vector coop[i] = cooperation rate of a homogeneous population of strategy i.
In the small-mutation limit the long-run abundance of any subset S of U only
depends on rho restricted to S, so one matrix per (model, regime, dynamics)
serves every strategy set in the multiverse.

Compatible with numpy >= 1.26 (no numpy-2-only API).
"""
import itertools
import numpy as np
from scipy.stats import hypergeom

# ----------------------------------------------------------------------------
# Fixation probabilities in a two-strategy population of size N
# ----------------------------------------------------------------------------

def fixation_from_payoffs(piA, piB, N, dyn):
    """Fixation probability of a single A mutant in a B resident population.

    piA[..., j-1], piB[..., j-1] are the expected payoffs of A and B players
    when the population holds j A players, j = 1..N-1 (leading axes broadcast).
    dyn = dict(kind='fermi', beta=...) or dict(kind='moran', w=..., pmin=..., pmax=...).
    """
    piA = np.asarray(piA, dtype=float)
    piB = np.asarray(piB, dtype=float)
    if dyn["kind"] == "fermi":
        logr = -dyn["beta"] * (piA - piB)          # log T-/T+
    elif dyn["kind"] == "moran":
        span = dyn["pmax"] - dyn["pmin"]
        fA = 1.0 + dyn["w"] * (piA - dyn["pmin"]) / span
        fB = 1.0 + dyn["w"] * (piB - dyn["pmin"]) / span
        if np.any(fA <= 0) or np.any(fB <= 0):
            raise ValueError("non-positive fitness")
        logr = np.log(fB) - np.log(fA)
    else:
        raise ValueError(dyn["kind"])
    cs = np.cumsum(logr, axis=-1)
    terms = np.concatenate([np.zeros(cs.shape[:-1] + (1,)), cs], axis=-1)
    m = terms.max(axis=-1, keepdims=True)
    lse = m[..., 0] + np.log(np.exp(terms - m).sum(axis=-1))
    return np.exp(-lse)


def pair_payoffs_linear(A, N):
    """piI[i,k,j-1], piK[i,k,j-1]: payoffs of an i-player and a k-player when the
    population holds j i-players and N-j k-players (pairwise game, no self-play)."""
    j = np.arange(1, N, dtype=float)
    aii = np.diag(A)[:, None, None]
    akk = np.diag(A)[None, :, None]
    aik = A[:, :, None]
    aki = A.T[:, :, None]
    piI = ((j - 1) * aii + (N - j) * aik) / (N - 1)
    piK = (j * aki + (N - j - 1) * akk) / (N - 1)
    return piI, piK



def rho_for_dyns(piI, piK, N, dyns, pmin, pmax):
    """Fixation matrices for several update rules from one payoff computation."""
    out = {}
    for name, dyn in dyns.items():
        d = dict(dyn)
        if d["kind"] == "moran":
            d.setdefault("pmin", pmin)
            d.setdefault("pmax", pmax)
        rho = fixation_from_payoffs(piI, piK, N, d)
        np.fill_diagonal(rho, 0.0)
        out[name] = rho
    return out

# ----------------------------------------------------------------------------
# Small-mutation-limit stationary distribution
# ----------------------------------------------------------------------------

def _gth_py(Q):
    """Grassmann-Taksar-Heyman elimination for the stationary vector of an
    irreducible continuous-time chain with off-diagonal rates Q[i, j] (i -> j).
    Subtraction-free, hence stable for rates spanning many orders of magnitude."""
    A = np.array(Q, dtype=float)
    k = A.shape[0]
    np.fill_diagonal(A, 0.0)
    for n in range(k - 1, 0, -1):
        s = A[n, :n].sum()
        A[:n, n] /= s
        A[:n, :n] += np.outer(A[:n, n], A[n, :n])
    pi = np.zeros(k)
    pi[0] = 1.0
    for n in range(1, k):
        pi[n] = pi[:n] @ A[:n, n]
    return pi / pi.sum()


try:  # numba-accelerated version (same algorithm)
    import numba as _nb

    @_nb.njit(cache=True)
    def _gth_nb(Q):
        k = Q.shape[0]
        A = Q.copy()
        for i in range(k):
            A[i, i] = 0.0
        for n in range(k - 1, 0, -1):
            s = 0.0
            for j in range(n):
                s += A[n, j]
            for i in range(n):
                A[i, n] /= s
            for i in range(n):
                ain = A[i, n]
                if ain != 0.0:
                    for j in range(n):
                        A[i, j] += ain * A[n, j]
        pi = np.zeros(k)
        pi[0] = 1.0
        for n in range(1, k):
            t = 0.0
            for i in range(n):
                t += pi[i] * A[i, n]
            pi[n] = t
        tot = pi.sum()
        for i in range(k):
            pi[i] /= tot
        return pi
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

RATE_FLOOR = 1e-300


def sml_stationary(rho_sub, kernel=None):
    """Stationary distribution of the embedded chain among homogeneous states.

    rho_sub[a, b] = fixation probability of a single a mutant in resident b.
    The transition rate from b to a is M(a|b) rho[a, b], where M is the
    mutation kernel restricted to the set: uniform (kernel=None) or
    proportional to kernel[a, b] and normalised over a != b for each b.
    Rates below RATE_FLOOR are raised to it so that the chain stays
    irreducible (the floor is never reached in the audited models).
    Solved with the GTH algorithm."""
    k = rho_sub.shape[0]
    if k == 1:
        return np.ones(1)
    Q = np.ascontiguousarray(rho_sub.T, dtype=float)    # Q[b, a] = rate b -> a
    if kernel is not None:
        W = np.array(kernel.T, dtype=float)              # W[b, a] = weight of a given resident b
        np.fill_diagonal(W, 0.0)
        W /= W.sum(axis=1, keepdims=True)
        Q = Q * W
    Q = np.maximum(Q, RATE_FLOOR)
    if HAVE_NUMBA:
        return _gth_nb(Q)
    return _gth_py(Q)


def sml_stationary_lu(rho_sub):
    """Previous (submitted) solver, retained only for the numerical comparison."""
    k = rho_sub.shape[0]
    if k == 1:
        return np.ones(1)
    Q = np.maximum(rho_sub.T, 1e-300)
    Q = Q.copy()
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    A = Q.T.copy()
    A[-1, :] = 1.0
    rhs = np.zeros(k)
    rhs[-1] = 1.0
    try:
        pi = np.linalg.solve(A, rhs)
        if not np.all(np.isfinite(pi)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        pi = np.linalg.lstsq(A, rhs, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


# ----------------------------------------------------------------------------
# Family DR: repeated prisoner's dilemma with memory-one strategies
# ----------------------------------------------------------------------------
# Strategy = (p0, pCC, pCD, pDC, pDD): cooperation probability in round 1 and
# after each outcome of the previous round (own action listed first).

DR_NAMES = {
    "ALLC": (1, 1, 1, 1, 1), "ALLD": (0, 0, 0, 0, 0), "TFT": (1, 1, 0, 1, 0),
    "STFT": (0, 1, 0, 1, 0), "WSLS": (1, 1, 0, 0, 1), "GRIM": (1, 1, 0, 0, 0),
}


def dr_universe(level):
    """'reactive' deterministic reactive strategies, deterministic first move (8);
    'm1det' deterministic memory-one strategies, deterministic first move (32);
    'm1grid' memory-one strategies with all five probabilities in {0, 1/2, 1} (243)."""
    if level == "reactive":
        return [(p0, pc, pd, pc, pd) for p0, pc, pd in itertools.product([1, 0], [1, 0], [1, 0])]
    if level == "m1det":
        return [tuple(s) for s in itertools.product([1, 0], repeat=5)]
    if level == "m1grid":
        return [tuple(s) for s in itertools.product([1, 0.5, 0], repeat=5)]
    raise ValueError(level)


def _code(x):
    return {1.0: "1", 0.0: "0", 0.5: "h"}[float(x)]


def dr_label(s):
    for k, v in DR_NAMES.items():
        if tuple(float(x) for x in v) == tuple(float(x) for x in s):
            return k
    return "".join(_code(x) for x in s)


def dr_payoff_matrix(strats, RSTP, delta, eps):
    """Normalised discounted payoff A[i,k] and cooperation rate C[i,k] of i against k.
    Execution errors flip each intended action with probability eps."""
    S = np.asarray(strats, dtype=float)
    S = (1 - eps) * S + eps * (1 - S)
    n = len(S)
    p0 = S[:, 0]
    p = S[:, 1:]
    q = p[:, [0, 2, 1, 3]]                  # co-player's view of the state
    pi_ = np.broadcast_to(p[:, None, :], (n, n, 4))
    qk = np.broadcast_to(q[None, :, :], (n, n, 4))
    M = np.stack([pi_ * qk, pi_ * (1 - qk), (1 - pi_) * qk, (1 - pi_) * (1 - qk)], axis=-1)
    a0, b0 = np.broadcast_arrays(p0[:, None], p0[None, :])
    v0 = np.stack([a0 * b0, a0 * (1 - b0), (1 - a0) * b0, (1 - a0) * (1 - b0)], axis=-1)
    Mt = np.swapaxes(np.eye(4)[None, None] - delta * M, -1, -2)
    v = np.linalg.solve(Mt.reshape(-1, 4, 4), ((1 - delta) * v0).reshape(-1, 4, 1)).reshape(n, n, 4)
    A = v @ np.asarray(RSTP, dtype=float)
    C = v[..., 0] + v[..., 1]
    return A, C


def dr_model(strats, RSTP=(3, 0, 4, 1), delta=0.9, eps=0.0, N=20, dyns=None):
    A, C = dr_payoff_matrix(strats, RSTP, delta, eps)
    piI, piK = pair_payoffs_linear(A, N)
    rho = rho_for_dyns(piI, piK, N, dyns, float(min(RSTP)), float(max(RSTP)))
    neutral = np.isclose(piI, piK, atol=1e-11, rtol=0).all(axis=-1)
    return dict(rho=rho, coop=np.diag(C).copy(), A=A, neutral=neutral, N=N, piI=piI, piK=piK,
                pbounds=(float(min(RSTP)), float(max(RSTP))))


# ----------------------------------------------------------------------------
# Family PU: optional public goods game with peer punishment
# ----------------------------------------------------------------------------
# Strategy = (action, pC, pD, pL): first-stage action in {'C','D','L'}; pX = 1
# if the strategy punishes every other group member whose first-stage action is X.

def pu_universe(level):
    """'hauert4' C, D, L and P (= C punishing D)                          (4)
    'gt9'   participants may punish C and/or D; loners neither punish nor are punished (9)
    'lone12' as gt9 but loners may also punish participants; loners are never punished (12)
    'rn24'  everyone may punish everyone                                   (24)"""
    if level == "hauert4":
        return [("C", 0, 0, 0), ("D", 0, 0, 0), ("L", 0, 0, 0), ("C", 0, 1, 0)]
    if level == "gt9":
        return [(a, pc, pd, 0) for a in "CD" for pc in (0, 1) for pd in (0, 1)] + [("L", 0, 0, 0)]
    if level == "lone12":
        return [(a, pc, pd, 0) for a in "CDL" for pc in (0, 1) for pd in (0, 1)]
    if level == "rn24":
        return [(a, pc, pd, pl) for a in "CDL" for pc in (0, 1) for pd in (0, 1) for pl in (0, 1)]
    raise ValueError(level)


def pu_label(s):
    a, pc, pd, pl = s
    tgt = "".join(x for x, f in zip("CDL", (pc, pd, pl)) if f)
    if (a, tgt) == ("C", "D"):
        return "P"
    return a if not tgt else f"{a}p{tgt}"


def _group_payoff(focal, other, m_other_focal, n, r, c, sigma, fine, cost, punish_if_game=False):
    """Payoff of a focal player in a group of n with m further players of its own
    strategy and n-1-m players of strategy `other`."""
    m_f = 1 + m_other_focal
    m_o = n - m_f
    counts = {"C": 0, "D": 0, "L": 0}
    counts[focal[0]] += m_f
    counts[other[0]] += m_o
    S = counts["C"] + counts["D"]
    a = focal[0]
    if a == "L" or S < 2:
        base = sigma
    else:
        base = r * c * counts["C"] / S - (c if a == "C" else 0.0)
    pidx = {"C": 1, "D": 2, "L": 3}
    targets = dict(counts)
    targets[a] -= 1
    given = sum(targets[x] for x in "CDL" if focal[pidx[x]])
    recv = (m_f - 1) * focal[pidx[a]] + m_o * other[pidx[a]]
    if punish_if_game and S < 2:      # sensitivity variant: no punishment stage without a game
        given, recv = 0, 0
    return base - cost * given - fine * recv


def pu_model(strats, n=5, r=3.0, c=1.0, sigma=1.0, fine=1.0, cost=0.3, N=100, dyns=None, punish_if_game=False):
    K = len(strats)
    g = np.zeros((K, K, n))
    for i, si in enumerate(strats):
        for k, sk in enumerate(strats):
            for m in range(n):
                g[i, k, m] = _group_payoff(si, sk, m, n, r, c, sigma, fine, cost, punish_if_game)
    j = np.arange(1, N)
    m = np.arange(n)
    Hi = hypergeom.pmf(m[None, :], N - 1, (j - 1)[:, None], n - 1)   # own-type co-players of an i player
    Hk = hypergeom.pmf(m[None, :], N - 1, j[:, None], n - 1)         # i co-players of a k player
    piI = np.einsum("jm,ikm->ikj", Hi, g)
    piK = np.einsum("jm,kim->ikj", Hk, g[:, :, ::-1])
    rho = rho_for_dyns(piI, piK, N, dyns, float(g.min()), float(g.max()))
    neutral = np.isclose(piI, piK, atol=1e-11, rtol=0).all(axis=-1)
    coop = np.array([1.0 if s[0] == "C" else 0.0 for s in strats])
    return dict(rho=rho, coop=coop, neutral=neutral, g=g, N=N, piI=piI, piK=piK,
                pbounds=(float(g.min()), float(g.max())))


# ----------------------------------------------------------------------------
# Family IR: indirect reciprocity, public institution, binary reputations
# ----------------------------------------------------------------------------
# Action rule p = (pGG, pGB, pBG, pBB): cooperation probability given (own
# reputation, recipient reputation).  Second-order norms map (action, recipient
# reputation) to a new reputation for the donor.

NORMS = {  # keys (action, recipient reputation) with 1 = C / good
    "SJ": {(1, 1): 1, (0, 1): 0, (1, 0): 0, (0, 0): 1},
    "SS": {(1, 1): 1, (0, 1): 0, (1, 0): 1, (0, 0): 1},
    "IS": {(1, 1): 1, (0, 1): 0, (1, 0): 1, (0, 0): 0},
    "SH": {(1, 1): 1, (0, 1): 0, (1, 0): 0, (0, 0): 0},
}

IR_NAMES = {"ALLC": (1, 1, 1, 1), "ALLD": (0, 0, 0, 0), "DISC": (1, 0, 1, 0), "pDISC": (0, 1, 0, 1)}


def ir_universe(level):
    """'recip4' rules conditioned on the recipient's reputation only (4);
    'own16' deterministic rules conditioned on own and recipient reputation (16);
    'grid81' as own16 with probabilities in {0, 1/2, 1} (81)."""
    if level == "recip4":
        return [(a, b, a, b) for a in (1, 0) for b in (1, 0)]
    if level == "own16":
        return [tuple(s) for s in itertools.product([1, 0], repeat=4)]
    if level == "grid81":
        return [tuple(s) for s in itertools.product([1, 0.5, 0], repeat=4)]
    raise ValueError(level)


def ir_label(s):
    for k, v in IR_NAMES.items():
        if tuple(float(x) for x in v) == tuple(float(x) for x in s):
            return k
    return "".join(_code(x) for x in s)


def ir_reputations(P, x, norm, eps, e, grid=51, bisect=48, root="largest"):
    """Mean-field equilibrium probability g[s] that an s-player has a good
    reputation; P (..., K, 4) action rules, x (..., K) frequencies.

    For a given average reputation gbar each type's reputation is the stationary
    probability of a two-state Markov chain, so equilibria are roots of the scalar
    map F(gbar) = sum_s x_s g_s(gbar) - gbar on [0, 1].  We select the largest
    root (the equilibrium reached from an all-good start when F is monotone),
    located on a grid of spacing 1/(grid-1) and refined by bisection."""
    d = np.array([[norm[(0, 0)], norm[(0, 1)]], [norm[(1, 0)], norm[(1, 1)]]], dtype=float)
    dt = (1 - e) * d + e * (1 - d)
    Pa = (1 - eps) * P
    pG = Pa[..., [1, 0]]
    pB = Pa[..., [3, 2]]
    # linear coefficients: P(good next | own state) = alpha + beta * gbar
    def coef(pc):
        q0 = pc[..., 0] * dt[1][0] + (1 - pc[..., 0]) * dt[0][0]   # recipient bad
        q1 = pc[..., 1] * dt[1][1] + (1 - pc[..., 1]) * dt[0][1]   # recipient good
        return q0, q1 - q0
    aA, aB = coef(pG)
    bA, bB = coef(pB)

    def g_of(gb):  # gb (..., 1) or (..., G)
        a = aA[..., None] + aB[..., None] * gb[..., None, :]
        b = bA[..., None] + bB[..., None] * gb[..., None, :]
        den = 1 - a + b
        return np.where(den > 0, b / np.where(den > 0, den, 1.0), 1.0)   # (..., K, G)

    def F(gb):
        return (x[..., None] * g_of(gb)).sum(axis=-2) - gb
    shp = x.shape[:-1]
    if root == "largest":
        ts = np.linspace(1.0, 0.0, grid)
        Fg = F(np.broadcast_to(ts, shp + (grid,)))
        nonneg = Fg >= -1e-15
        first = np.argmax(nonneg, axis=-1)            # first grid point (from top) with F >= 0
        hi = ts[np.maximum(first - 1, 0)]
        lo = ts[first]
        exact = first == 0
        lo = np.where(exact, 1.0, lo)
        hi = np.where(exact, 1.0, hi)
        for _ in range(bisect):
            mid = 0.5 * (lo + hi)
            fm = F(mid[..., None])[..., 0]
            pos = fm >= 0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        gstar = 0.5 * (lo + hi)
    else:  # smallest root: first grid point (from below) with F <= 0
        ts = np.linspace(0.0, 1.0, grid)
        Fg = F(np.broadcast_to(ts, shp + (grid,)))
        nonpos = Fg <= 1e-15
        first = np.argmax(nonpos, axis=-1)
        lo = ts[np.maximum(first - 1, 0)]
        hi = ts[first]
        exact = first == 0
        lo = np.where(exact, 0.0, lo)
        hi = np.where(exact, 0.0, hi)
        for _ in range(bisect):
            mid = 0.5 * (lo + hi)
            fm = F(mid[..., None])[..., 0]
            pos = fm > 0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        gstar = 0.5 * (lo + hi)
    return g_of(gstar[..., None])[..., 0]


def ir_root_count(P, x, norm, eps, e, grid=1001):
    """Number of sign changes of the scalar reputation map on a fine grid
    (diagnostic for multiple mean-field equilibria)."""
    d = np.array([[norm[(0, 0)], norm[(0, 1)]], [norm[(1, 0)], norm[(1, 1)]]], dtype=float)
    dt = (1 - e) * d + e * (1 - d)
    Pa = (1 - eps) * P
    pG = Pa[..., [1, 0]]
    pB = Pa[..., [3, 2]]
    def coef(pc):
        q0 = pc[..., 0] * dt[1][0] + (1 - pc[..., 0]) * dt[0][0]
        q1 = pc[..., 1] * dt[1][1] + (1 - pc[..., 1]) * dt[0][1]
        return q0, q1 - q0
    aA, aB = coef(pG)
    bA, bB = coef(pB)
    ts = np.linspace(0.0, 1.0, grid)
    a = aA[..., None] + aB[..., None] * ts
    b = bA[..., None] + bB[..., None] * ts
    den = 1 - a + b
    g = np.where(den > 0, b / np.where(den > 0, den, 1.0), 1.0)
    Fv = (x[..., None] * g).sum(axis=-2) - ts
    s = np.sign(np.where(np.abs(Fv) < 1e-14, 0.0, Fv))
    # count strict sign changes, treating exact zeros as roots
    ch = (s[..., 1:] * s[..., :-1] < 0).sum(axis=-1) + (s[..., 1:-1] == 0).sum(axis=-1)
    return ch


def _ir_give(Pa, g_s, g_rec):
    return (g_s * (g_rec * Pa[..., 0] + (1 - g_rec) * Pa[..., 1])
            + (1 - g_s) * (g_rec * Pa[..., 2] + (1 - g_rec) * Pa[..., 3]))


def _ir_coefs(P, nrm, eps, e):
    d = np.array([[nrm[(0, 0)], nrm[(0, 1)]], [nrm[(1, 0)], nrm[(1, 1)]]], dtype=float)
    dt = (1 - e) * d + e * (1 - d)
    Pa = (1 - eps) * P
    def coef(pc):
        q0 = pc[..., 0] * dt[1][0] + (1 - pc[..., 0]) * dt[0][0]
        q1 = pc[..., 1] * dt[1][1] + (1 - pc[..., 1]) * dt[0][1]
        return q0, q1 - q0
    aA, aB = coef(Pa[..., [1, 0]])
    bA, bB = coef(Pa[..., [3, 2]])
    return aA, aB, bA, bB


def ir_reputations_selfexcl(Pi, Pk, j, N, nrm, eps, e, iters=3000, tol=1e-13):
    """Two-type reputations when every donor meets a recipient drawn from the
    other N-1 individuals (self excluded in assessment).  Damped fixed-point
    iteration from the all-good state."""
    ai = _ir_coefs(Pi, nrm, eps, e)
    ak = _ir_coefs(Pk, nrm, eps, e)
    gi = np.ones(Pi.shape[:-1])
    gk = np.ones(Pk.shape[:-1])
    def f(c, gb):
        a = c[0] + c[1] * gb
        b = c[2] + c[3] * gb
        return b / (1 - a + b)
    for _ in range(iters):
        gbi = ((j - 1) * gi + (N - j) * gk) / (N - 1)
        gbk = (j * gi + (N - j - 1) * gk) / (N - 1)
        ni, nk = f(ai, gbi), f(ak, gbk)
        if max(np.max(np.abs(ni - gi)), np.max(np.abs(nk - gk))) < tol:
            gi, gk = ni, nk
            break
        gi, gk = 0.5 * (gi + ni), 0.5 * (gk + nk)
    return gi, gk


def ir_model(strats, norm="SJ", b=3.0, c=1.0, eps=0.02, e=0.02, N=50, dyns=None, root="largest", selfx="hybrid"):
    """Indirect reciprocity under a public institution.

    selfx='hybrid' (pre-specified): reputations from the mean-field map with the
    recipient drawn from the whole population; donations received exclude the
    focal individual.  selfx='full': recipients are drawn from the other N-1
    individuals in assessment, donating and receiving."""
    P = np.asarray(strats, dtype=float)
    K = len(P)
    nrm = NORMS[norm]
    j = np.arange(1, N, dtype=float)
    Pi = np.broadcast_to(P[:, None, None, :], (K, K, N - 1, 4))
    Pk = np.broadcast_to(P[None, :, None, :], (K, K, N - 1, 4))
    Pai, Pak = (1 - eps) * Pi, (1 - eps) * Pk
    xi = j / N
    if selfx == "hybrid":
        PP = np.stack([Pi, Pk], axis=-2)
        X = np.broadcast_to(np.stack([xi, 1 - xi], axis=-1)[None, None], (K, K, N - 1, 2))
        g = ir_reputations(PP, X, nrm, eps, e, root=root)
        gi, gk = g[..., 0], g[..., 1]
        gbar_i = gbar_k = xi * gi + (1 - xi) * gk
    else:
        gi, gk = ir_reputations_selfexcl(Pi, Pk, j, N, nrm, eps, e)
        gbar_i = ((j - 1) * gi + (N - j) * gk) / (N - 1)
        gbar_k = (j * gi + (N - j - 1) * gk) / (N - 1)
    give_i = _ir_give(Pai, gi, gbar_i)
    give_k = _ir_give(Pak, gk, gbar_k)
    piI = b * (((j - 1) * _ir_give(Pai, gi, gi) + (N - j) * _ir_give(Pak, gk, gi)) / (N - 1)) - c * give_i
    piK = b * ((j * _ir_give(Pai, gi, gk) + (N - j - 1) * _ir_give(Pak, gk, gk)) / (N - 1)) - c * give_k
    rho = rho_for_dyns(piI, piK, N, dyns, -float(c), float(b))
    neutral = np.isclose(piI, piK, atol=1e-11, rtol=0).all(axis=-1)
    gh = ir_reputations(P[:, None, :], np.ones((K, 1)), nrm, eps, e, root=root)[:, 0]
    coop = _ir_give((1 - eps) * P, gh, gh)
    return dict(rho=rho, coop=coop, neutral=neutral, rep=gh, N=N, piI=piI, piK=piK,
                pbounds=(-float(c), float(b)))
