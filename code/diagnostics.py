"""Numerical diagnostics added at revision.

R1  multiplicity of mean-field reputation equilibria: for every ordered pair of the
    81 indirect-reciprocity rules and every two-strategy composition (N = 50), the
    number of roots of the scalar reputation map on a 1001-point grid, under stern
    judging and image scoring, b irrelevant (reputations do not depend on b).
R2  stationary solver: GTH against 120-digit arithmetic (mpmath) for the complete
    81-rule stern-judging chains under strong selection (beta = 10), and the
    submitted LU solver for comparison.
R3  deterministic unit checks: neutral fixation = 1/N; fixation probabilities in
    [0, 1]; stationary vectors non-negative, normalised, balance residual; clone
    invariance of the quotient (duplicating a strategy label changes the labelled
    result but not the quotient result).
Usage: python diagnostics.py OUT.json
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E
import audit2 as A2

OUT = {}
# ---------------------------------------------------------------- R1
U = np.array(E.ir_universe("grid81"), float)
K = len(U)
N = 50
j = np.arange(1, N)
xi = j / N
res = {}
for norm in ("SJ", "IS"):
    tot = 0
    multi = 0
    worst = 0
    for i in range(K):
        Pi = np.broadcast_to(U[i], (K, N - 1, 4))
        Pk = np.broadcast_to(U[:, None, :], (K, N - 1, 4))
        PP = np.stack([Pi, Pk], axis=-2)
        X = np.broadcast_to(np.stack([xi, 1 - xi], -1), (K, N - 1, 2))
        cnt = E.ir_root_count(PP, X, E.NORMS[norm], 0.02, 0.02)
        tot += cnt.size
        multi += int((cnt > 1).sum())
        worst = max(worst, int(cnt.max()))
    homo = E.ir_root_count(U[:, None, :], np.ones((K, 1)), E.NORMS[norm], 0.02, 0.02)
    res[norm] = dict(compositions=tot, with_multiple_roots=multi, max_roots=worst,
                     homogeneous_multiple=int((homo > 1).sum()))
OUT["R1_root_multiplicity"] = res

# ---------------------------------------------------------------- R2
try:
    import mpmath as mp
    mp.mp.dps = 120
    r2 = []
    for b in (3.0, 4.0, 5.0):
        m = E.ir_model(E.ir_universe("grid81"), norm="SJ", b=b, N=50, dyns={"D2": {"kind": "fermi", "beta": 10.0}})
        rho = m["rho"]["D2"]
        pg = E.sml_stationary(rho)
        pl = E.sml_stationary_lu(rho)
        # high-precision GTH
        Q = [[mp.mpf(max(float(rho[a, bb]), 1e-300)) if a != bb else mp.mpf(0) for a in range(K)] for bb in range(K)]
        for n in range(K - 1, 0, -1):
            s = mp.fsum(Q[n][:n])
            for i in range(n):
                Q[i][n] = Q[i][n] / s
            for i in range(n):
                if Q[i][n] != 0:
                    for jj in range(n):
                        Q[i][jj] += Q[i][n] * Q[n][jj]
        pi = [mp.mpf(1)] + [mp.mpf(0)] * (K - 1)
        for n in range(1, K):
            pi[n] = mp.fsum(pi[i] * Q[i][n] for i in range(n))
        tot = mp.fsum(pi)
        ph = np.array([float(p / tot) for p in pi])
        r2.append(dict(b=b, coop_gth=float(pg @ m["coop"]), coop_hp=float(ph @ m["coop"]), coop_lu=float(pl @ m["coop"]),
                       max_abs_err_gth=float(np.abs(pg - ph).max()), max_abs_err_lu=float(np.abs(pl - ph).max())))
    OUT["R2_solver"] = r2
except ImportError:
    OUT["R2_solver"] = "mpmath not available"

# ---------------------------------------------------------------- R3
chk = {}
m = E.dr_model(E.dr_universe("m1det"), delta=0.9, eps=0.0, N=20,
               dyns={"F": {"kind": "fermi", "beta": 1.0}, "M": {"kind": "moran", "w": 1.0}})
neu = m["neutral"]
for d in ("F", "M"):
    rho = m["rho"][d]
    off = ~np.eye(len(rho), dtype=bool)
    chk[f"neutral_fix_maxdev_{d}"] = float(np.abs(rho[neu & off] - 1 / 20).max())
    chk[f"rho_range_{d}"] = [float(rho[off].min()), float(rho[off].max())]
    pi = E.sml_stationary(rho)
    Qm = rho.T.copy()
    np.fill_diagonal(Qm, 0)
    np.fill_diagonal(Qm, -Qm.sum(1))
    chk[f"stationary_residual_{d}"] = float(np.abs(pi @ Qm).max())
    chk[f"stationary_min_{d}"] = float(pi.min())
# clone invariance toy: triad + duplicated ALLC label
tri = [E.DR_NAMES[x] for x in ("ALLC", "TFT", "ALLD")]
dup = tri + [E.DR_NAMES["ALLC"]]
for name, S in (("triad", tri), ("triad_plus_ALLC_clone", dup)):
    mm = E.dr_model(S, delta=0.9, eps=0.0, N=20, dyns={"D0": {"kind": "fermi", "beta": 1.0}})
    pi = E.sml_stationary(mm["rho"]["D0"])
    cl = A2.clone_classes(mm, range(len(S)))
    reps = sorted(set(cl.values()))
    piq = E.sml_stationary(mm["rho"]["D0"][np.ix_(reps, reps)])
    chk[f"clone_{name}"] = dict(labelled_coop=float(pi @ mm["coop"]), quotient_coop=float(piq @ mm["coop"][reps]),
                                n_classes=len(reps))
OUT["R3_unit"] = chk
json.dump(OUT, open(sys.argv[1] if len(sys.argv) > 1 else "diagnostics.json", "w"), indent=1)
print(json.dumps(OUT)[:3000])
