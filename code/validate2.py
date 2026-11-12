"""Independent validation of the audit engine.

V1  payoffs: Monte Carlo repeated games (DR), explicit group simulation (PU),
    agent-based public-reputation dynamics (IR) against the engine.
V2  fixation probabilities: Monte Carlo of the two-strategy birth-death chain
    against the closed form.
V3  small-mutation limit: exact stationary distribution of the full
    finite-mutation imitation process against the SML abundances.
Writes validation.json.
"""
import itertools
import json
import os
import sys
from math import comb
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

rng = np.random.default_rng(20261105)
OUT = {}

# ------------------------------------------------------------------ V1 DR --
def mc_repeated(p, q, RSTP, delta, eps, ngames=40000):
    p = (1 - eps) * np.asarray(p, float) + eps * (1 - np.asarray(p, float))
    q = (1 - eps) * np.asarray(q, float) + eps * (1 - np.asarray(q, float))
    pay = np.array(RSTP, float)
    tot = np.zeros(ngames)
    alive = np.ones(ngames, bool)
    a = rng.random(ngames) < p[0]
    b = rng.random(ngames) < q[0]
    while alive.any():
        st = (~a).astype(int) * 2 + (~b).astype(int)          # 0 CC,1 CD,2 DC,3 DD
        tot[alive] += pay[st[alive]]
        alive &= rng.random(ngames) < delta
        sb = np.array([0, 2, 1, 3])[st]
        a = rng.random(ngames) < p[1:][st]
        b = rng.random(ngames) < q[1:][sb]
    est = (1 - delta) * tot
    return est.mean(), est.std() / np.sqrt(ngames)

U = E.dr_universe("m1grid")
pairs = [tuple(rng.choice(len(U), 2, replace=False)) for _ in range(12)]
A, _ = E.dr_payoff_matrix(U, (3, 0, 4, 1), 0.9, 0.01)
res = []
for i, k in pairs:
    m, se = mc_repeated(U[i], U[k], (3, 0, 4, 1), 0.9, 0.01)
    res.append(dict(i=E.dr_label(U[i]), k=E.dr_label(U[k]), exact=float(A[i, k]), mc=float(m), se=float(se),
                    z=float((m - A[i, k]) / se)))
OUT["V1_DR"] = res

# ------------------------------------------------------------------ V1 PU --
def naive_group(strats_in_group, r, c, sigma, fine, cost):
    acts = [s[0] for s in strats_in_group]
    S = sum(a in "CD" for a in acts)
    nC = sum(a == "C" for a in acts)
    pay = []
    for idx, s in enumerate(strats_in_group):
        if s[0] == "L" or S < 2:
            u = sigma
        else:
            u = r * c * nC / S - (c if s[0] == "C" else 0)
        for jdx, t in enumerate(strats_in_group):
            if jdx == idx:
                continue
            tgt = {"C": 1, "D": 2, "L": 3}
            if s[tgt[t[0]]]:
                u -= cost
            if t[tgt[s[0]]]:
                u -= fine
        pay.append(u)
    return pay

Up = E.pu_universe("rn24")
N, n = 30, 5
r_, c_, sig, fine, cost = 3.0, 1.0, 1.0, 1.0, 0.3
g = np.zeros((24, 24, n))
for i, si in enumerate(Up):
    for k, sk in enumerate(Up):
        for m in range(n):
            g[i, k, m] = E._group_payoff(si, sk, m, n, r_, c_, sig, fine, cost)
from scipy.stats import hypergeom
res = []
for _ in range(10):
    i, k = rng.choice(24, 2, replace=False)
    j = int(rng.integers(1, N))
    exact = float(sum(hypergeom.pmf(m, N - 1, j - 1, n - 1) * g[i, k, m] for m in range(n)))
    popn = [Up[i]] * j + [Up[k]] * (N - j)
    vals = []
    for _ in range(20000):
        others = rng.choice(np.arange(1, N), n - 1, replace=False)      # focal = individual 0 (an i player)
        grp = [popn[0]] + [popn[o] for o in others]
        vals.append(naive_group(grp, r_, c_, sig, fine, cost)[0])
    vals = np.array(vals)
    res.append(dict(i=E.pu_label(Up[i]), k=E.pu_label(Up[k]), j=j, exact=exact, mc=float(vals.mean()),
                    se=float(vals.std() / np.sqrt(len(vals))),
                    z=float((vals.mean() - exact) / max(vals.std() / np.sqrt(len(vals)), 1e-12))))
OUT["V1_PU"] = res

# ------------------------------------------------------------------ V1 IR --
def abm_reputation(P, counts, norm, eps, e, b, c, steps=2_200_000, burn=200_000):
    nrm = E.NORMS[norm]
    types = np.repeat(np.arange(len(P)), counts)
    Np = len(types)
    rep = np.ones(Np, int)
    P = np.asarray(P, float)
    gsum = np.zeros(len(P)); pay = np.zeros(Np); cnt = 0
    donors = rng.integers(0, Np, steps)
    recips = rng.integers(0, Np - 1, steps)
    recips = recips + (recips >= donors)
    u1 = rng.random(steps); u2 = rng.random(steps); u3 = rng.random(steps)
    for t in range(steps):
        d, rr = donors[t], recips[t]
        pr = P[types[d], (1 - rep[d]) * 2 + (1 - rep[rr])]
        act = 1 if (u1[t] < pr and u2[t] >= eps) else 0
        newrep = nrm[(act, int(rep[rr]))]
        if u3[t] < e:
            newrep = 1 - newrep
        if t >= burn:
            pay[d] -= c * act
            pay[rr] += b * act
        rep[d] = newrep
        if t >= burn and t % 50 == 0:
            for s in range(len(P)):
                gsum[s] += rep[types == s].mean()
            cnt += 1
    rounds = (steps - burn) / Np
    pay_type = np.array([pay[types == s].mean() / rounds for s in range(len(P))])
    return gsum / cnt, pay_type

res = []
for norm, rules, counts in [("SJ", [(1, 0, 1, 0), (0, 0, 0, 0)], [35, 15]),
                            ("SJ", [(1, 0, 1, 0), (1, 1, 1, 1)], [25, 25]),
                            ("IS", [(1, 0, 1, 0), (0, 0, 0, 0)], [30, 20]),
                            ("SJ", [(1, 0.5, 0, 1), (1, 0, 1, 0)], [20, 30])]:
    x = np.array(counts, float) / sum(counts)
    Np = sum(counts)
    P2 = np.asarray(rules, float)
    gab, payab = abm_reputation(rules, counts, norm, 0.02, 0.02, 3.0, 1.0)
    Pa = 0.98 * P2
    # production (pre-specified) convention: mean-field reputations with recipients drawn
    # from the whole population; donations received exclude the focal individual
    gmf = E.ir_reputations(P2, x, E.NORMS[norm], 0.02, 0.02)
    gbar = float(x @ gmf)
    give = E._ir_give(Pa, gmf, gbar)
    recv = np.array([sum((counts[t] - (t == s)) / (Np - 1) * E._ir_give(Pa[t], gmf[t], gmf[s]) for t in range(2))
                     for s in range(2)])
    pay_prod = 3.0 * recv - 1.0 * give
    # fully self-excluded convention (revision sensitivity variant)
    j0 = float(counts[0])
    gi, gk = E.ir_reputations_selfexcl(P2[0][None], P2[1][None], np.array([j0]), Np, E.NORMS[norm], 0.02, 0.02)
    gfx = np.array([gi[0], gk[0]])
    gb0 = ((j0 - 1) * gfx[0] + (Np - j0) * gfx[1]) / (Np - 1)
    gb1 = (j0 * gfx[0] + (Np - j0 - 1) * gfx[1]) / (Np - 1)
    givefx = np.array([E._ir_give(Pa[0], gfx[0], gb0), E._ir_give(Pa[1], gfx[1], gb1)])
    recvfx = np.array([sum((counts[t] - (t == s)) / (Np - 1) * E._ir_give(Pa[t], gfx[t], gfx[s]) for t in range(2))
                       for s in range(2)])
    pay_fx = 3.0 * recvfx - 1.0 * givefx
    res.append(dict(norm=norm, rules=[E.ir_label(r) for r in rules], counts=counts,
                    g_meanfield=gmf.tolist(), g_selfexcl=gfx.tolist(), g_abm=gab.tolist(),
                    pay_meanfield=pay_prod.tolist(), pay_selfexcl=pay_fx.tolist(), pay_abm=payab.tolist()))
OUT["V1_IR"] = res

# ------------------------------------------------------------------ V2 -----
def mc_fixation(piA, piB, N, beta, runs=20000):
    piA = np.asarray(piA); piB = np.asarray(piB)
    tp = (N - np.arange(1, N)) / N * np.arange(1, N) / N * 1 / (1 + np.exp(-beta * (piA - piB)))
    tm = (N - np.arange(1, N)) / N * np.arange(1, N) / N * 1 / (1 + np.exp(beta * (piA - piB)))
    pup = tp / (tp + tm)
    fix = 0
    for _ in range(runs):
        j = 1
        while 0 < j < N:
            j += 1 if rng.random() < pup[j - 1] else -1
        fix += j == N
    return fix / runs

res = []
Ud = E.dr_universe("m1det")
A0, _ = E.dr_payoff_matrix(Ud, (3, 0, 4, 1), 0.9, 0.0)
lab = [E.dr_label(s) for s in Ud]
for a, bname in [("ALLD", "TFT"), ("TFT", "ALLD"), ("ALLC", "TFT"), ("ALLD", "ALLC"), ("WSLS", "ALLD"), ("GRIM", "ALLC")]:
    i, k = lab.index(a), lab.index(bname)
    piI, piK = E.pair_payoffs_linear(A0[np.ix_([i, k], [i, k])], 20)
    exact = float(E.fixation_from_payoffs(piI[0, 1], piK[0, 1], 20, {"kind": "fermi", "beta": 1.0}))
    mc = mc_fixation(piI[0, 1], piK[0, 1], 20, 1.0)
    res.append(dict(mutant=a, resident=bname, exact=exact, mc=mc, se=float(np.sqrt(max(mc * (1 - mc), 1e-9) / 20000))))
OUT["V2"] = res

# ------------------------------------------------------------------ V3 -----
def compositions(N, K):
    for cut in itertools.combinations(range(N + K - 1), K - 1):
        prev = -1; out = []
        for cc in cut + (N + K - 1,):
            out.append(cc - prev - 1); prev = cc
        yield tuple(out)

def exact_finite_mu(payfun, K, N, beta, mu):
    states = np.array(list(compositions(N, K)))
    index = {tuple(s): i for i, s in enumerate(states)}
    PI = payfun(states)                                   # (S, K) payoff of each type (nan if absent)
    rows, cols, vals = [], [], []
    for si, s in enumerate(states):
        for a in range(K):
            if s[a] == 0:
                continue
            pa = s[a] / N
            for b2 in range(K):
                if b2 == a:
                    continue
                rate = pa * mu / (K - 1)
                if s[b2] > 0:
                    rate += pa * (1 - mu) * s[b2] / (N - 1) / (1 + np.exp(-beta * (PI[si, b2] - PI[si, a])))
                t = s.copy(); t[a] -= 1; t[b2] += 1
                rows.append(si); cols.append(index[tuple(t)]); vals.append(rate)
    S = len(states)
    T = sp.csr_matrix((vals, (rows, cols)), shape=(S, S))
    out = np.asarray(T.sum(axis=1)).ravel()
    Q = (T - sp.diags(out)).T.tolil()
    Q[0, :] = 1.0
    rhs = np.zeros(S); rhs[0] = 1.0
    pi = spla.spsolve(Q.tocsr(), rhs)
    pi = np.clip(pi, 0, None); pi /= pi.sum()
    freq = (pi[:, None] * states / N).sum(axis=0)
    return freq

def dr_payfun(Asub, N):
    def f(states):
        return (states @ Asub.T - np.diag(Asub)[None, :]) / (N - 1)
    return f

def ir_payfun(rules, norm, b, c, N):
    P = np.asarray(rules, float)
    def f(states):
        x = states / N
        g = E.ir_reputations(np.broadcast_to(P, (len(states),) + P.shape), x, E.NORMS[norm], 0.02, 0.02)
        Pa = 0.98 * P
        gbar = (x * g).sum(axis=1, keepdims=True)
        give = E._ir_give(Pa[None], g, gbar)
        out = np.zeros(states.shape)
        for s in range(len(P)):
            recv = np.zeros(len(states))
            for t in range(len(P)):
                w = (states[:, t] - (t == s)) / (N - 1)
                recv += w * E._ir_give(Pa[t][None], g[:, t], g[:, s])
            out[:, s] = b * recv - c * give[:, s]
        return out
    return f

def pu_payfun(strats, N, n, r, c, sigma, fine, cost):
    K = len(strats)
    combos = list(compositions(n - 1, K))
    tab = np.zeros((K, len(combos)))
    for a in range(K):
        for ci, cmb in enumerate(combos):
            grp = [strats[a]] + [strats[t] for t in range(K) for _ in range(cmb[t])]
            tab[a, ci] = naive_group(grp, r, c, sigma, fine, cost)[0]
    combos = np.array(combos)
    def f(states):
        out = np.zeros(states.shape)
        for a in range(K):
            avail = states - (np.arange(K) == a)
            pr = np.ones((len(states), len(combos)))
            for t in range(K):
                pr *= np.vectorize(comb)(np.maximum(avail[:, t:t + 1], 0), combos[None, :, t])
            pr /= comb(N - 1, n - 1)
            out[:, a] = pr @ tab[a]
        return out
    return f

res = []
for mu in (1e-2, 1e-3, 1e-4):
    # DR triad N=20, delta=0.9
    tri = [E.DR_NAMES[x] for x in ("ALLC", "TFT", "ALLD")]
    Asub, Csub = E.dr_payoff_matrix(tri, (3, 0, 4, 1), 0.9, 0.0)
    m = E.dr_model(tri, (3, 0, 4, 1), 0.9, 0.0, 20, {"D0": {"kind": "fermi", "beta": 1.0}})
    sml = E.sml_stationary(m["rho"]["D0"])
    fr = exact_finite_mu(dr_payfun(Asub, 20), 3, 20, 1.0, mu)
    res.append(dict(model="DR triad N=20", mu=mu, sml=sml.tolist(), finite=fr.tolist(),
                    coop_sml=float(sml @ m["coop"]), coop_finite_approx=float(fr @ m["coop"])))
    # IR triad SJ b=3 N=50
    rules = [E.IR_NAMES[x] for x in ("ALLC", "ALLD", "DISC")]
    mi = E.ir_model(rules, norm="SJ", b=3.0, N=50, dyns={"D0": {"kind": "fermi", "beta": 1.0}})
    sml = E.sml_stationary(mi["rho"]["D0"])
    fr = exact_finite_mu(ir_payfun(rules, "SJ", 3.0, 1.0, 50), 3, 50, 1.0, mu)
    res.append(dict(model="IR SJ triad N=50", mu=mu, sml=sml.tolist(), finite=fr.tolist(),
                    coop_sml=float(sml @ mi["coop"]), coop_finite_approx=float(fr @ mi["coop"])))
    # PU Hauert 4 N=40
    strats = E.pu_universe("hauert4")
    mp = E.pu_model(strats, N=40, cost=0.3, dyns={"D0": {"kind": "fermi", "beta": 1.0}})
    sml = E.sml_stationary(mp["rho"]["D0"])
    fr = exact_finite_mu(pu_payfun(strats, 40, 5, 3.0, 1.0, 1.0, 1.0, 0.3), 4, 40, 1.0, mu)
    res.append(dict(model="PU Hauert-4 N=40", mu=mu, sml=sml.tolist(), finite=fr.tolist(),
                    coop_sml=float(sml @ mp["coop"]), coop_finite_approx=float(fr @ mp["coop"])))
    print("V3 mu", mu, flush=True)
OUT["V3"] = res

json.dump(OUT, open(sys.argv[1] if len(sys.argv) > 1 else "validation.json", "w"), indent=1)
print("ok")
