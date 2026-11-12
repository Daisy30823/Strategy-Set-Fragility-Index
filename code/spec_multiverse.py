"""Specification multiverse for the authors' strategy sets (revision analysis).

For every claim-regime of the frozen corpus, the authors' set S_A is held fixed and
300 evolutionary specifications are drawn at random: update rule (pairwise
comparison or Moran birth-death with linear fitness, probability 1/2 each),
selection strength (beta or w log-uniform on [0.1, 10]) and population size
(uniform integer on [N0/2, 2 N0]).  The specification fragility index (SpFI) is
the share of draws whose verdict differs from the primary-specification verdict.
In addition, the authors'-set verdict is evaluated under finite mutation
(mu = 1e-3 and 1e-2) by solving the full stationary distribution exactly.

Usage: python spec_multiverse.py OUT.json SHARD NSHARDS
"""
import itertools
import json
import os
import sys
import zlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E
import claims as CL
import audit2 as A2

NSPEC = 300


def build_SA(claim, regime, N, dyns):
    fam = claim["family"]
    S = [A2.name_tuple(fam, n) for n in claim["SA"]]
    if fam == "DR":
        m = E.dr_model(S, RSTP=CL.DR_BASE["RSTP"], delta=regime["delta"], eps=regime["eps"], N=N, dyns=dyns)
    elif fam == "PU":
        m = E.pu_model(S, n=CL.PU_BASE["n"], r=regime["r"], c=CL.PU_BASE["c"], sigma=CL.PU_BASE["sigma"],
                       fine=regime["fine"], cost=regime["cost"], N=N, dyns=dyns)
    else:
        m = E.ir_model(S, norm=regime["norm"], b=regime["b"], c=CL.IR_BASE["c"], eps=CL.IR_BASE["eps"],
                       e=CL.IR_BASE["e"], N=N, dyns=dyns)
    m["U"] = [A2.canon(s) for s in S]
    m["N"] = N
    return m


def verdict_SA(claim, regime, N, dyn, name="x"):
    dyns = {name: dyn}
    mm = {"main": build_SA(claim, regime, N, dyns)}
    if claim.get("arms") == "norm_SJ_vs_IS":
        alt = dict(regime)
        alt["norm"] = "IS"
        mm["alt"] = build_SA(claim, alt, N, dyns)
    idx_of = {s: i for i, s in enumerate(mm["main"]["U"])}
    ev = A2.Evaluator(claim, mm, name, idx_of)
    mg = ev.margin(list(range(len(idx_of))))[0]
    return bool(ev.verdict(mg)), float(mg)


# ------------------------------------------------- finite-mutation exact chain
def compositions(N, K):
    for cut in itertools.combinations(range(N + K - 1), K - 1):
        prev = -1
        out = []
        for cc in cut + (N + K - 1,):
            out.append(cc - prev - 1)
            prev = cc
        yield out


def finite_mu(claim, regime, mu, beta=1.0):
    """Stationary composition distribution of the pairwise-comparison process with
    uniform mutation probability mu among the strategies of S_A.  Returns the
    long-run cooperation (population average of the pairwise cooperation rates)
    and strategy abundances."""
    fam = claim["family"]
    S = [A2.name_tuple(fam, n) for n in claim["SA"]]
    K = len(S)
    N = A2.base_N(fam)
    states = np.array(list(compositions(N, K)))
    index = {tuple(s): i for i, s in enumerate(states)}
    x = states / N
    if fam == "DR":
        A, C = E.dr_payoff_matrix(S, CL.DR_BASE["RSTP"], regime["delta"], regime["eps"])
        PI = (states @ A.T - np.diag(A)[None, :]) / (N - 1)
        CO = ((states @ C.T - np.diag(C)[None, :]) / (N - 1))  # cooperation rate of each type
    elif fam == "IR":
        P = np.asarray(S, float)
        nrm = E.NORMS[regime["norm"]]
        g = E.ir_reputations(np.broadcast_to(P, (len(states),) + P.shape), x, nrm, CL.IR_BASE["eps"], CL.IR_BASE["e"])
        Pa = (1 - CL.IR_BASE["eps"]) * P
        gbar = (x * g).sum(1, keepdims=True)
        give = E._ir_give(Pa[None], g, gbar)
        PI = np.zeros(states.shape)
        for s in range(K):
            recv = np.zeros(len(states))
            for t in range(K):
                w = (states[:, t] - (t == s)) / (N - 1)
                recv += w * E._ir_give(Pa[t][None], g[:, t], g[:, s])
            PI[:, s] = regime["b"] * recv - CL.IR_BASE["c"] * give[:, s]
        CO = give
    else:
        from math import comb
        n = CL.PU_BASE["n"]
        combos = np.array(list(compositions(n - 1, K)))
        tab = np.zeros((K, len(combos)))
        for a in range(K):
            for ci, cmb in enumerate(combos):
                # payoff of a focal a-player whose co-players have composition cmb
                grp_counts = {"C": 0, "D": 0, "L": 0}
                grp = [S[a]] + [S[t] for t in range(K) for _ in range(cmb[t])]
                for s_ in grp:
                    grp_counts[s_[0]] += 1
                Sn = grp_counts["C"] + grp_counts["D"]
                me = S[a]
                base = (CL.PU_BASE["sigma"] if (me[0] == "L" or Sn < 2) else
                                   regime["r"] * CL.PU_BASE["c"] * grp_counts["C"] / Sn - (CL.PU_BASE["c"] if me[0] == "C" else 0))
                pidx = {"C": 1, "D": 2, "L": 3}
                given = sum(1 for s_ in grp[1:] if me[pidx[s_[0]]])
                recv = sum(1 for s_ in grp[1:] if s_[pidx[me[0]]])
                tab[a, ci] = base - regime["cost"] * given - regime["fine"] * recv
        PI = np.zeros(states.shape)
        from scipy.special import comb as scomb
        for a in range(K):
            avail = states - (np.arange(K) == a)
            pr = np.ones((len(states), len(combos)))
            for t in range(K):
                pr *= scomb(np.maximum(avail[:, t:t + 1], 0), combos[None, :, t])
            pr /= scomb(N - 1, n - 1)
            PI[:, a] = pr @ tab[a]
        CO = np.tile(np.array([1.0 if s_[0] == "C" else 0.0 for s_ in S]), (len(states), 1))
    rows, cols, vals = [], [], []
    for si, s in enumerate(states):
        for a in range(K):
            if s[a] == 0:
                continue
            pa = s[a] / N
            for b in range(K):
                if b == a:
                    continue
                rate = pa * mu / (K - 1)
                if s[b] > 0:
                    rate += pa * (1 - mu) * s[b] / (N - 1) / (1 + np.exp(-beta * (PI[si, b] - PI[si, a])))
                t = s.copy()
                t[a] -= 1
                t[b] += 1
                rows.append(si)
                cols.append(index[tuple(t)])
                vals.append(rate)
    nS = len(states)
    T = sp.csr_matrix((vals, (rows, cols)), shape=(nS, nS))
    outr = np.asarray(T.sum(axis=1)).ravel()
    Q = (T - sp.diags(outr)).T.tolil()
    Q[0, :] = 1.0
    rhs = np.zeros(nS)
    rhs[0] = 1.0
    pi = spla.spsolve(Q.tocsr(), rhs)
    pi = np.clip(pi, 0, None)
    pi /= pi.sum()
    abund = (pi[:, None] * x).sum(0)
    coop = float(pi @ (x * CO).sum(1))
    return coop, abund.tolist()


def finite_mu_verdict(claim, regime, mu):
    t = claim["type"]
    if t in ("level", "control"):
        coop, ab = finite_mu(claim, regime, mu)
        mg = claim["sign"] * (coop - claim["theta"])
        return bool(mg > A2.TOL), mg, coop
    if t == "winner":
        coop, ab = finite_mu(claim, regime, mu)
        names = claim["SA"]
        mg = ab[names.index(claim["X"])] - max(ab[names.index(r)] for r in claim["rivals"])
        return bool(mg > A2.TOL), mg, coop
    return None, None, None  # stability and comparative claims are not defined at finite mu here


def run(args):
    claim, ri = args
    regime = claim["regimes"][ri]
    fam = claim["family"]
    N0 = A2.base_N(fam)
    v0, m0 = verdict_SA(claim, regime, N0, {"kind": "fermi", "beta": 1.0})
    rng = np.random.default_rng(zlib.crc32(f"spec|{claim['id']}|{ri}".encode()))
    draws = []
    for _ in range(NSPEC):
        rule = "fermi" if rng.random() < 0.5 else "moran"
        s = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        N = int(rng.integers(N0 // 2, 2 * N0 + 1))
        dyn = {"kind": "fermi", "beta": s} if rule == "fermi" else {"kind": "moran", "w": s}
        v, mg = verdict_SA(claim, regime, N, dyn)
        draws.append(dict(rule=rule, strength=s, N=N, verdict=v, margin=mg))
    spfi = float(np.mean([d["verdict"] != v0 for d in draws]))
    fm = {}
    if claim["type"] in ("level", "control", "winner") and fam != "PU":
        for mu in (1e-3, 1e-2):
            try:
                fm[str(mu)] = finite_mu_verdict(claim, regime, mu)
            except Exception as ex:  # pragma: no cover
                fm[str(mu)] = ["error", str(ex)]
    return dict(claim=claim["id"], family=fam, type=claim["type"], regime=ri, verdict_D0=v0, margin_D0=m0,
                spfi=spfi, n_spec=NSPEC, finite_mu=fm,
                spfi_fermi_only=float(np.mean([d["verdict"] != v0 for d in draws if d["rule"] == "fermi"])),
                spfi_moran_only=float(np.mean([d["verdict"] != v0 for d in draws if d["rule"] == "moran"])),
                draws=draws)


if __name__ == "__main__":
    out = sys.argv[1]
    shard, nsh = int(sys.argv[2]), int(sys.argv[3])
    tasks = [(c, ri) for c in CL.CLAIMS for ri in range(len(c["regimes"]))][shard::nsh]
    res = [run(t) for t in tasks]
    json.dump(res, open(out, "w"))
    print("ok", len(res))
