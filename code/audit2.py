"""Revised multiverse audit (revision round 1).

Changes relative to the submitted audit:
  * stationary distributions by the GTH algorithm (engine.sml_stationary);
  * seeds keyed on the strategy-set specification (identical universes give identical draws);
  * exact fragility for stability claims (superset 1-2^-q, replacement hypergeometric);
  * Clopper-Pearson intervals for sampled SSFI;
  * size-stratified SSFI(k), inclusion-probability sensitivity SSFI(q), single additions;
  * SML-equivalence (clone) quotient universes;
  * verdict-reversal Banzhaf index and margin Shapley values with Monte Carlo standard errors;
  * two revision-stage specifications (D6 local mutation kernel, D7 common N = 50);
  * model variants (IR root selection, IR self-exclusion, PU punishment only when a game is played).

Usage: python audit2.py OUTDIR NPROC [--variant NAME] [--only ID,ID] [--light] [--exploratory]
"""
import argparse
import itertools
import json
import math
import os
import sys
import zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E
import claims as CL

TOL = 1e-9
NDRAW = 20000
NQ = 4000
NK = 2000
QS = (0.05, 0.1, 0.25, 0.75)
NPERM = 1000
VARIANTS = {
    "primary": {},
    "ir_smallest_root": {"root": "smallest"},
    "ir_full_selfexcl": {"selfx": "full"},
    "pu_punish_if_game": {"punish_if_game": True},
}


def canon(s):
    return tuple(float(x) if not isinstance(x, str) else x for x in s)


def universe_tuples(family, level):
    f = {"DR": E.dr_universe, "PU": E.pu_universe, "IR": E.ir_universe}[family]
    return [canon(s) for s in f(level)]


def name_tuple(family, name):
    d = {"DR": CL.DR_S, "PU": CL.PU_S, "IR": CL.IR_S}[family]
    return canon(d[name])


def label(family, s):
    return {"DR": E.dr_label, "PU": E.pu_label, "IR": E.ir_label}[family](s)


def base_N(family):
    return {"DR": CL.DR_BASE["N"], "PU": CL.PU_BASE["N"], "IR": CL.IR_BASE["N"]}[family]


def spec_N(family, spec):
    if spec["N"] is not None:
        return int(spec["N"])
    return int(round(base_N(family) * spec["mult"]))


# ------------------------------------------------------------------ models --
def build(family, regime, N, dyns, variant):
    v = VARIANTS[variant]
    if family == "DR":
        U = E.dr_universe("m1grid")
        m = E.dr_model(U, RSTP=CL.DR_BASE["RSTP"], delta=regime["delta"], eps=regime["eps"], N=N, dyns=dyns)
    elif family == "PU":
        U = E.pu_universe("rn24")
        m = E.pu_model(U, n=CL.PU_BASE["n"], r=regime["r"], c=CL.PU_BASE["c"], sigma=CL.PU_BASE["sigma"],
                       fine=regime["fine"], cost=regime["cost"], N=N, dyns=dyns,
                       punish_if_game=v.get("punish_if_game", False))
    else:
        U = E.ir_universe("grid81")
        m = E.ir_model(U, norm=regime["norm"], b=regime["b"], c=CL.IR_BASE["c"], eps=CL.IR_BASE["eps"],
                       e=CL.IR_BASE["e"], N=N, dyns=dyns, root=v.get("root", "largest"), selfx=v.get("selfx", "hybrid"))
    m["U"] = [canon(s) for s in U]
    m["N"] = N
    return m


def local_kernel(family, U):
    """Mutation weights w[a, b] = exp(-lambda d(a, b)): L1 distance in cooperation
    probabilities (DR, IR; lambda = 3) or Hamming distance over action and
    punishment targets (PU; lambda = 1.5)."""
    if family == "PU":
        X = np.array([[{"C": 0, "D": 1, "L": 2}[s[0]], s[1], s[2], s[3]] for s in U], dtype=float)
        d = (X[:, None, 0] != X[None, :, 0]).astype(float) + np.abs(X[:, None, 1:] - X[None, :, 1:]).sum(-1)
        lam = 1.5
    else:
        X = np.array(U, dtype=float)
        d = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
        lam = 3.0
    return np.exp(-lam * d)


def clone_classes(m, idxs, tol=1e-10):
    """SML-equivalence classes within the strategy indices idxs: i ~ j iff the
    two strategies are payoff-neutral with each other in every composition,
    have identical payoffs against and for every other member of the set, and
    the same homogeneous cooperation rate.  Returns a dict index -> class id."""
    piI, piK, coop = m["piI"], m["piK"], m["coop"]
    idxs = list(idxs)
    parent = {i: i for i in idxs}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    neu = m["neutral"]
    arr = np.array(idxs)
    for a_pos, i in enumerate(idxs):
        for j in idxs[a_pos + 1:]:
            if not (neu[i, j] and neu[j, i]) or abs(coop[i] - coop[j]) > tol:
                continue
            others = arr[(arr != i) & (arr != j)]
            if (np.abs(piI[i, others] - piI[j, others]).max(initial=0) < tol and
                    np.abs(piK[i, others] - piK[j, others]).max(initial=0) < tol and
                    np.abs(piI[others, i] - piI[others, j]).max(initial=0) < tol and
                    np.abs(piK[others, i] - piK[others, j]).max(initial=0) < tol):
                parent[find(j)] = find(i)
    return {i: find(i) for i in idxs}


# ----------------------------------------------------------------- margins --
class Evaluator:
    def __init__(self, claim, models, dname, idx_of, kernel=None, quotient_ctrl=None):
        self.c = claim
        self.m = models
        self.dn = dname
        self.idx_of = idx_of
        self.kernel = kernel
        self.N = models["main"]["N"]
        fam = claim["family"]
        if claim["type"] == "winner":
            self.X = idx_of[name_tuple(fam, claim["X"])]
            self.riv = [idx_of[name_tuple(fam, r)] for r in claim["rivals"]]
        if claim["type"] == "stability":
            self.X = idx_of[name_tuple(fam, claim["focal"])]
        if claim.get("arms") == "drop_loners":
            self.is_loner = np.array([s[0] == "L" for s in models["main"]["U"]])
        self.qctrl = quotient_ctrl  # optional map index -> representative for the compulsory arm

    def coop(self, S, arm="main"):
        S = np.asarray(S)
        rho = self.m[arm]["rho"][self.dn][np.ix_(S, S)]
        K = None if self.kernel is None else self.kernel[np.ix_(S, S)]
        pi = E.sml_stationary(rho, K)
        return float(pi @ self.m[arm]["coop"][S]), pi

    def margin(self, S, full=False):
        t = self.c["type"]
        S = np.asarray(sorted(set(int(x) for x in S)))
        if t in ("level", "control"):
            cval, _ = self.coop(S)
            return self.c["sign"] * (cval - self.c["theta"]), cval
        if t == "winner":
            cval, pi = self.coop(S)
            pos = {s: i for i, s in enumerate(S)}
            mg = pi[pos[self.X]] - max(pi[pos[r]] for r in self.riv)
            if full:  # strict reading: X ahead of every other member of S
                others = [pi[i] for s, i in pos.items() if s != self.X]
                return mg, cval, pi[pos[self.X]] - max(others)
            return mg, cval
        if t == "comparative":
            if self.c["arms"] == "drop_loners":
                ca, _ = self.coop(S)
                Sb = S[~self.is_loner[S]]
                if self.qctrl is not None:
                    Sb = np.array(sorted({self.qctrl[int(x)] for x in Sb}))
                cb, _ = self.coop(Sb)
            else:
                ca, _ = self.coop(S, "main")
                cb, _ = self.coop(S, "alt")
            return ca - cb, ca, cb
        if t == "stability":
            others = [s for s in S if s != self.X]
            mx = self.m["main"]["rho"][self.dn][others, self.X].max() if others else 0.0
            return 1.0 / self.N - mx, np.nan
        raise ValueError(t)

    def verdict(self, mg):
        if self.c["type"] == "stability":
            return mg >= -1e-12
        return mg > TOL


def cp_interval(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def seed_of(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode())


# ---------------------------------------------------------------- one task --
def run_task(args):
    claim, ri, outdir, variant, light = args
    regime = claim["regimes"][ri]
    fam = claim["family"]
    rows = []
    dnames = list(CL.DYNAMICS) if variant == "primary" else ["D0"]
    # group specifications by population size
    byN = {}
    for dn in dnames:
        sp = CL.DYNAMICS[dn]
        byN.setdefault(spec_N(fam, sp), {})[dn] = sp["dyn"]
    models = {}
    for N, dyns in byN.items():
        mm = {"main": build(fam, regime, N, dyns, variant)}
        if claim.get("arms") == "norm_SJ_vs_IS":
            alt = dict(regime)
            alt["norm"] = "IS"
            mm["alt"] = build(fam, alt, N, dyns, variant)
        for dn in dyns:
            models[dn] = mm
    base = models["D0"]["main"]
    Umax = base["U"]
    idx_of = {s: i for i, s in enumerate(Umax)}
    KL = local_kernel(fam, Umax)
    SA = [idx_of[name_tuple(fam, n)] for n in claim["SA"]]
    core = [idx_of[name_tuple(fam, n)] for n in claim["core"]]
    focal = idx_of[name_tuple(fam, claim["focal"])]
    done_levels = {}
    for ui, ulev in enumerate(claim["U"]):
        utag = CL.UNIV_TAGS[ui]
        if ulev in done_levels:  # identical universe: copy rows, flag duplicate
            for r in done_levels[ulev]:
                rr = dict(r)
                rr["universe"] = utag
                rr["dup_of"] = r["universe"]
                rows.append(rr)
            continue
        Uset = [idx_of[s] for s in universe_tuples(fam, ulev)]
        omitted = sorted(set(Uset) - set(SA))
        mo = len(omitted)
        om = np.array(omitted)
        rho0, N0, neu = base["rho"]["D0"], base["N"], base["neutral"]
        Z = np.array(Uset)
        n_neutral = n_entry = 0
        flux = 0.0
        cats = {}
        for y in omitted:
            ne = bool(neu[y, focal] and neu[focal, y])
            en = bool(ne and np.any((rho0[Z, y] > 1.0 / N0 + 1e-12) & (rho0[Z, focal] <= 1.0 / N0 + 1e-12)))
            n_neutral += ne
            n_entry += en
            cats[y] = "entry" if en else ("neutral" if ne else "other")
            flux += rho0[y, focal] * N0 * max(0.0, float(np.max(rho0[Z, y] - rho0[Z, focal]))) * N0
        # clone quotient of the universe (D0 payoffs; payoff equality is dynamics independent)
        cls = clone_classes(base, Uset)
        rep_of_class = {}
        for s in SA:
            rep_of_class.setdefault(cls[s], s)
        for s in Uset:
            rep_of_class.setdefault(cls[s], s)
        Uq = sorted(set(rep_of_class.values()))
        omq = sorted(set(Uq) - {rep_of_class[cls[s]] for s in SA})
        qctrl = None
        if claim.get("arms") == "drop_loners":
            nonL = [s for s in Uset if Umax[s][0] != "L"]
            cl2 = clone_classes(base, nonL)
            r2 = {}
            for s in nonL:
                r2.setdefault(cl2[s], s)
            qctrl = {s: r2[cl2[s]] for s in nonL}
        urows = []
        for dn in dnames:
            sp = CL.DYNAMICS[dn]
            K = KL if sp["kernel"] == "local" else None
            ev = Evaluator(claim, models[dn], dn, idx_of, kernel=K)
            res_A = ev.margin(SA)
            res_U = ev.margin(Uset)
            mg_A, mg_U = res_A[0], res_U[0]
            v_A, v_U = ev.verdict(mg_A), ev.verdict(mg_U)
            row = dict(claim=claim["id"], family=fam, type=claim["type"], regime=ri, regime_par=regime,
                       variant=variant, universe=utag, ulevel=ulev, U_size=len(Uset), n_omitted=mo, dyn=dn,
                       N=models[dn]["main"]["N"], margin_A=float(mg_A), coop_A=float(res_A[1]), verdict_A=bool(v_A),
                       margin_U=float(mg_U), coop_U=float(res_U[1]), verdict_U=bool(v_U),
                       n_neutral=n_neutral, n_entry=n_entry, entry_flux=float(flux),
                       U_q_size=len(Uq), n_omitted_q=len(omq), dup_of=None)
            if claim["type"] == "comparative":
                row["ctrl_A"], row["ctrl_U"] = float(res_A[2]), float(res_U[2])
            if claim["type"] == "winner":
                fA = ev.margin(SA, full=True)[2]
                fU = ev.margin(Uset, full=True)[2]
                row["strict_margin_A"], row["strict_margin_U"] = float(fA), float(fU)
            rng = np.random.default_rng(seed_of(claim["id"], ri, ulev, dn, variant))
            # ---------------- superset SSFI
            stab = claim["type"] == "stability"
            if stab:
                rho_d = models[dn]["main"]["rho"][dn]
                X = ev.X
                inv = [y for y in omitted if rho_d[y, X] > 1.0 / ev.N + 1e-12]
                qinv = len(inv)
                row["n_invaders"] = qinv
                if v_A:
                    row["ssfi"] = 1.0 - 2.0 ** (-qinv)
                else:  # authors' set not reproduced: SSFI undefined, keep for completeness
                    row["ssfi"] = float("nan")
                row["ssfi_exact"] = True
                row["n_sets"] = 2 ** mo if mo <= 60 else None
                row["ssfi_ci"] = [row["ssfi"], row["ssfi"]]
            else:
                if mo <= 16:
                    masks = np.array(list(itertools.product([0, 1], repeat=mo)), dtype=bool).reshape(-1, mo)
                    exact = True
                else:
                    masks = rng.random((NDRAW, mo)) < 0.5
                    exact = False
                mgs = np.empty(len(masks))
                cvs = np.empty(len(masks))
                for kk, mk in enumerate(masks):
                    r_ = ev.margin(SA + list(om[mk]))
                    mgs[kk], cvs[kk] = r_[0], r_[1]
                ver = mgs > TOL
                rev = ver != v_A
                row["ssfi"] = float(rev.mean())
                row["ssfi_exact"] = exact
                row["n_sets"] = int(len(masks))
                row["ssfi_ci"] = [row["ssfi"], row["ssfi"]] if exact else list(cp_interval(int(rev.sum()), len(rev)))
                row["q05"], row["q50"], row["q95"] = (float(np.quantile(mgs, q)) for q in (0.05, 0.5, 0.95))
                row["coop_q50"] = float(np.nanmedian(cvs)) if np.isfinite(cvs).any() else None
            # ---------------- replacement SSFI
            pool = sorted(set(Uset) - set(core))
            kfree = len(SA) - len(core)
            row["ssfi_rep"] = None
            if kfree > 0 and len(pool) >= kfree:
                if stab:
                    qp = len([y for y in pool if models[dn]["main"]["rho"][dn][y, ev.X] > 1.0 / ev.N + 1e-12])
                    if v_A:
                        row["ssfi_rep"] = 1.0 - math.comb(len(pool) - qp, kfree) / math.comb(len(pool), kfree)
                else:
                    ncomb = math.comb(len(pool), kfree)
                    combos = (list(itertools.combinations(pool, kfree)) if ncomb <= NDRAW else
                              [tuple(rng.choice(pool, kfree, replace=False)) for _ in range(NDRAW)])
                    rv = [ev.verdict(ev.margin(core + list(cmb))[0]) for cmb in combos]
                    row["ssfi_rep"] = float(np.mean(np.array(rv) != v_A))
            # ---------------- clone quotient: complete set + superset SSFI
            evq = Evaluator(claim, models[dn], dn, idx_of, kernel=K, quotient_ctrl=qctrl)
            SAq = sorted({rep_of_class[cls[s]] for s in SA})
            rq = evq.margin(Uq)
            row["margin_Uq"], row["coop_Uq"], row["verdict_Uq"] = float(rq[0]), float(rq[1]), bool(evq.verdict(rq[0]))
            rqa = evq.margin(SAq)
            row["verdict_Aq"] = bool(evq.verdict(rqa[0]))
            omq_a = np.array(omq)
            if dn in ("D0",) or not light:
                if stab:
                    rho_d = models[dn]["main"]["rho"][dn]
                    qi = len([y for y in omq if rho_d[y, evq.X] > 1.0 / evq.N + 1e-12])
                    row["ssfi_q"] = (1.0 - 2.0 ** (-qi)) if row["verdict_Aq"] else float("nan")
                elif len(omq) <= 16:
                    mk_all = np.array(list(itertools.product([0, 1], repeat=len(omq))), dtype=bool).reshape(-1, len(omq))
                    vv = np.array([evq.verdict(evq.margin(SAq + list(omq_a[mk]))[0]) for mk in mk_all])
                    row["ssfi_q"] = float(np.mean(vv != row["verdict_Aq"]))
                else:
                    rq_ = np.random.default_rng(seed_of(claim["id"], ri, ulev, dn, variant, "q"))
                    mk_all = rq_.random((NDRAW // 4, len(omq))) < 0.5
                    vv = np.array([evq.verdict(evq.margin(SAq + list(omq_a[mk]))[0]) for mk in mk_all])
                    row["ssfi_q"] = float(np.mean(vv != row["verdict_Aq"]))
            # ---------------- D0-only analyses
            if dn == "D0" and not light and mo > 0:
                # inclusion-probability sensitivity
                sq = {}
                for qv in QS:
                    if stab:
                        sq[str(qv)] = (1.0 - (1.0 - qv) ** row["n_invaders"]) if v_A else None
                        continue
                    rr = np.random.default_rng(seed_of(claim["id"], ri, ulev, dn, variant, "qs", qv))
                    mk_all = rr.random((NQ, mo)) < qv
                    vv = np.array([ev.verdict(ev.margin(SA + list(om[mk]))[0]) for mk in mk_all])
                    sq[str(qv)] = float(np.mean(vv != v_A))
                sq["0.5"] = row["ssfi"]
                row["ssfi_by_q"] = sq
                # size-stratified
                ks = sorted({k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k < mo} | {mo})
                sk = {}
                for kv in ks:
                    if stab:
                        sk[str(kv)] = (1.0 - math.comb(mo - row["n_invaders"], kv) / math.comb(mo, kv)) if v_A else None
                        continue
                    nc = math.comb(mo, kv)
                    if nc <= NK:
                        sets = [list(cmb) for cmb in itertools.combinations(range(mo), kv)]
                    else:
                        rr = np.random.default_rng(seed_of(claim["id"], ri, ulev, dn, variant, "k", kv))
                        sets = [list(rr.choice(mo, kv, replace=False)) for _ in range(NK)]
                    vv = np.array([ev.verdict(ev.margin(SA + list(om[st]))[0]) for st in sets])
                    sk[str(kv)] = float(np.mean(vv != v_A))
                row["ssfi_by_k"] = sk
                # single additions that reverse
                singles = []
                for y in omitted:
                    if ev.verdict(ev.margin(SA + [y])[0]) != v_A:
                        singles.append(label(fam, Umax[y]))
                row["single_reversers"] = singles
                # attribution
                if not stab:
                    Xd = np.column_stack([np.ones(len(masks)), masks.astype(float)])
                    coef, *_ = np.linalg.lstsq(Xd, mgs, rcond=None)
                    banz = coef[1:]
                    inc = masks.astype(float)
                    n_in = inc.sum(0)
                    n_out = len(masks) - n_in
                    p_in = (inc * rev[:, None]).sum(0) / np.maximum(n_in, 1)
                    p_out = ((1 - inc) * rev[:, None]).sum(0) / np.maximum(n_out, 1)
                    vban = p_in - p_out
                    vse = np.sqrt(p_in * (1 - p_in) / np.maximum(n_in, 1) + p_out * (1 - p_out) / np.maximum(n_out, 1))
                else:
                    rho_d = models[dn]["main"]["rho"][dn]
                    excess = np.array([rho_d[y, ev.X] - 1.0 / ev.N for y in omitted])
                    banz = np.full(mo, np.nan)
                    vban = np.where(excess > 1e-12, 2.0 ** (-(row["n_invaders"] - 1)) if row["n_invaders"] else 0, 0.0)
                    vse = np.zeros(mo)
                # Shapley (margin), permutation sampling with standard errors
                nperm = NPERM
                contrib = np.zeros((nperm, mo))
                for p in range(nperm):
                    perm = rng.permutation(mo)
                    cur = list(SA)
                    prev = mg_A
                    for o in perm:
                        cur.append(om[o])
                        nxt = ev.margin(cur)[0]
                        contrib[p, o] = nxt - prev
                        prev = nxt
                shap = contrib.mean(0)
                shap_se = contrib.std(0, ddof=1) / np.sqrt(nperm)
                half = contrib[: nperm // 2].mean(0), contrib[nperm // 2:].mean(0)
                labs = np.array([label(fam, Umax[o]) for o in om])
                catv = np.array([cats[o] for o in om])
                row["shapley_sum"] = float(shap.sum())
                row["shapley_by_cat"] = {c: [int((catv == c).sum()), float(shap[catv == c].sum())] for c in ("entry", "neutral", "other")}
                order = np.argsort(-np.abs(shap))[:10]
                row["shapley_top"] = [[labs[o], float(shap[o]), float(shap_se[o]), catv[o]] for o in order]
                ordv = np.argsort(-np.abs(vban))[:10]
                row["vbanzhaf_top"] = [[labs[o], float(vban[o]), float(vse[o])] for o in ordv]
                from scipy.stats import spearmanr
                top = np.argsort(-np.abs(shap))[:20]
                row["shapley_split_rank_rho"] = float(spearmanr(half[0][top], half[1][top]).correlation) if mo > 2 else None
                np.savez_compressed(os.path.join(outdir, f"spec_{variant}_{claim['id']}_{ri}_{ulev}.npz"),
                                    margins=(mgs.astype(np.float32) if not stab else np.zeros(0)),
                                    coop=(cvs.astype(np.float32) if not stab else np.zeros(0)),
                                    size=(masks.sum(1).astype(np.int16) if not stab else np.zeros(0)),
                                    masks=(np.packbits(masks, axis=1) if not stab else np.zeros(0)),
                                    omitted=labs, cats=catv, shapley=shap, shapley_se=shap_se,
                                    banzhaf_margin=banz, banzhaf_verdict=vban, banzhaf_verdict_se=vse,
                                    margin_A=mg_A, margin_U=mg_U)
            urows.append(row)
        done_levels[ulev] = urows
        rows.extend(urows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("nproc", type=int)
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--only", default=None)
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    corpus = CL.EXPLORATORY if a.exploratory else CL.CLAIMS
    tasks = []
    for c in corpus:
        if a.only and c["id"] not in a.only.split(","):
            continue
        if a.variant.startswith("ir_") and c["family"] != "IR":
            continue
        if a.variant.startswith("pu_") and c["family"] != "PU":
            continue
        for ri in range(len(c["regimes"])):
            tasks.append((c, ri, a.outdir, a.variant, a.light))
    tasks.sort(key=lambda t: ({"DR": 0, "IR": 1, "PU": 2}[t[0]["family"]], t[0]["id"], t[1]))
    tasks = tasks[a.shard::a.nshards]
    fn = os.path.join(a.outdir, f"cells_{a.variant}{'_expl' if a.exploratory else ''}_s{a.shard}.jsonl")
    out = open(fn, "w")
    for t in tasks:
        rows = run_task(t)
        for r in rows:
            out.write(json.dumps(r, default=float) + "\n")
        out.flush()
        print("done", rows[0]["claim"], rows[0]["regime"], flush=True)
    out.close()


if __name__ == "__main__":
    main()
