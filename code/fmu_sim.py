"""Finite-mutation individual-based simulations in the largest direct-reciprocity
universes (revision analysis of the small-mutation-limit assumption).

Process: population of N; at each step a random focal individual mutates with
probability mu (to a strategy drawn uniformly from the other strategies of the
set) or otherwise imitates a random other individual with the Fermi probability.
Long-run cooperation is the time average of the population mean pairwise
cooperation rate.  Four independent replicates per case.

Usage: python fmu_sim.py OUT.json SHARD NSHARDS
"""
import json
import os
import sys
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E


@nb.njit(cache=True)
def simulate(A, C, N, beta, mu, steps, burn, seed, sample_every):
    np.random.seed(seed)
    K = A.shape[0]
    pop = np.zeros(N, dtype=np.int64)
    pop[:] = np.random.randint(K)
    tot = 0.0
    cnt = 0
    for t in range(steps):
        f = np.random.randint(N)
        if np.random.random() < mu:
            new = np.random.randint(K - 1)
            if new >= pop[f]:
                new += 1
            pop[f] = new
        else:
            mdl = np.random.randint(N - 1)
            if mdl >= f:
                mdl += 1
            sf = pop[f]
            sm = pop[mdl]
            if sf != sm:
                pf = 0.0
                pm = 0.0
                for i in range(N):
                    if i != f:
                        pf += A[sf, pop[i]]
                    if i != mdl:
                        pm += A[sm, pop[i]]
                pf /= N - 1
                pm /= N - 1
                if np.random.random() < 1.0 / (1.0 + np.exp(-beta * (pm - pf))):
                    pop[f] = sm
        if t >= burn and (t % sample_every) == 0:
            s = 0.0
            for i in range(N):
                for j in range(N):
                    if i != j:
                        s += C[pop[i], pop[j]]
            tot += s / (N * (N - 1))
            cnt += 1
    return tot / cnt


def case_list():
    out = []
    for delta, eps, lab in [(0.8, 0.0, "DR1 d=0.8"), (0.9, 0.0, "DR1 d=0.9"), (0.8, 0.01, "DR4 d=0.8")]:
        for lev in ("m1det", "m1grid"):
            for mu in (1e-2, 1e-3, 1e-4):
                out.append(dict(case=lab, delta=delta, eps=eps, universe=lev, mu=mu))
    return out


def run(case):
    U = E.dr_universe(case["universe"])
    A, C = E.dr_payoff_matrix(U, (3, 0, 4, 1), case["delta"], case["eps"])
    m = E.dr_model(U, delta=case["delta"], eps=case["eps"], N=20, dyns={"D0": {"kind": "fermi", "beta": 1.0}})
    pi = E.sml_stationary(m["rho"]["D0"])
    sml = float(pi @ m["coop"])
    steps = int(round(2000.0 / case["mu"] * 400))  # about 8e5 mutation events at mu=1e-2; 8e7 steps at 1e-4
    steps = min(steps, 800_000_000)
    reps = [float(simulate(A, C, 20, 1.0, case["mu"], steps, steps // 10, 1000 + r, 97)) for r in range(4)]
    return dict(case, sml=sml, sim=reps, sim_mean=float(np.mean(reps)), sim_se=float(np.std(reps, ddof=1) / 2), steps=steps)


if __name__ == "__main__":
    out, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    res = [run(c) for c in case_list()[shard::nsh]]
    json.dump(res, open(out, "w"), indent=1)
    print("ok")
