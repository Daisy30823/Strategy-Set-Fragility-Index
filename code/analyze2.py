"""Summaries, hypothesis tests and sensitivity analyses for the revised audit.

Usage: python analyze2.py RESULTS_DIR
Reads RESULTS_DIR/cells_*.jsonl, spec_*.json, fmu_*.json, diagnostics.json, validation2.json.
Writes RESULTS_DIR/summary2.json and CSV tables.
"""
import glob
import itertools
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claims as CL

RES = sys.argv[1] if len(sys.argv) > 1 else "rev_out"
rng = np.random.default_rng(20261105)


def load(pattern):
    fs = sorted(glob.glob(os.path.join(RES, pattern)))
    return pd.concat([pd.read_json(f, lines=True) for f in fs], ignore_index=True) if fs else pd.DataFrame()


C = load("cells_primary_s*.jsonl")
C["dup"] = C.dup_of.notna()
C["longrun"] = C.type.isin(["level", "winner", "comparative"])
d0 = C[C.dyn == "D0"].copy()
rp = d0[d0.verdict_A].copy()          # reproduced, all universe labels (identical sets repeated)
rpd = rp[~rp.dup].copy()               # reproduced, distinct strategy-set specifications
S = {}
S["counts"] = dict(cells=int(len(C)), distinct_cells=int((~C.dup).sum()), claim_regimes=int(d0.groupby(["claim", "regime"]).ngroups),
                   reproduced=int(rp.groupby(["claim", "regime"]).ngroups),
                   not_reproduced=[f"{c}-{r}" for (c, r), g in d0.groupby(["claim", "regime"]) if not g.verdict_A.iloc[0]],
                   exact_cells_D0=int(d0[~d0.dup].ssfi_exact.sum()), sampled_cells_D0=int((~d0[~d0.dup].ssfi_exact).sum()))

# --------------------------------------------------------------- field level
F = {}
for u in CL.UNIV_TAGS:
    x = rp[rp.universe == u]
    rev = x.groupby("claim").verdict_U.apply(lambda s: bool((~s).any()))
    revq = x.groupby("claim").verdict_Uq.apply(lambda s: bool((~s).any()))
    F[u] = dict(n=int(len(x)), ssfi_mean=float(x.ssfi.mean()), ssfi_median=float(x.ssfi.median()),
                cells_gt_0p1=int((x.ssfi > 0.1).sum()), cells_reversed=int((~x.verdict_U).sum()),
                claims_reversed=sorted(rev[rev].index), ssfi_q_mean=float(x.ssfi_q.mean()),
                cells_reversed_q=int((~x.verdict_Uq).sum()), claims_reversed_q=sorted(revq[revq].index),
                by_family={k: float(v) for k, v in x.groupby("family").ssfi.mean().items()},
                by_family_q={k: float(v) for k, v in x.groupby("family").ssfi_q.mean().items()},
                by_type={k: float(v) for k, v in x.groupby("type").ssfi.mean().items()},
                rep_mean=float(x.ssfi_rep.mean()), rep_n=int(x.ssfi_rep.notna().sum()),
                claim_mean_of_means=float(x.groupby("claim").ssfi.mean().mean()))
S["field"] = F
S["controls_max_ssfi"] = float(rp[rp.type == "control"].ssfi.max())
# sampled-zero upper bounds
z = rp[(~rp.ssfi_exact) & (rp.ssfi == 0)]
S["sampled_zero_cells"] = int(len(z))

# ------------------------------------------------------- clone quotient table
Q = rp[rp.U_q_size < rp.U_size][["claim", "regime", "universe", "U_size", "U_q_size", "coop_A", "coop_U", "coop_Uq",
                                 "verdict_U", "verdict_Uq", "ssfi", "ssfi_q"]]
Q.to_csv(os.path.join(RES, "quotient_table.csv"), index=False)
S["quotient_changed_cells"] = int(len(Q))

# ------------------------------------------------ size and inclusion sensitivity
mx = rp[(rp.universe == "U_max")]
qs = ["0.05", "0.1", "0.25", "0.5", "0.75"]
S["ssfi_by_q_mean"] = {q: float(np.nanmean([r[q] if r[q] is not None else np.nan for r in mx.ssfi_by_q])) for q in qs}
S["ssfi_by_q_claimmean"] = {q: float(mx.assign(v=[r[q] if r[q] is not None else np.nan for r in mx.ssfi_by_q]).groupby("claim").v.mean().mean()) for q in qs}
byk = []
for _, r in mx.iterrows():
    for k, v in r.ssfi_by_k.items():
        byk.append(dict(claim=r.claim, regime=r.regime, family=r.family, k=int(k), m=int(r.n_omitted), ssfi_k=v))
BK = pd.DataFrame(byk)
BK.to_csv(os.path.join(RES, "ssfi_by_k.csv"), index=False)
S["single_reversers"] = {f"{r.claim}-{r.regime}": len(r.single_reversers) for _, r in mx.iterrows()}
S["cells_with_single_reverser"] = int(sum(len(r.single_reversers) > 0 for _, r in mx.iterrows()))
S["frac_single_k1_mean"] = float(BK[BK.k == 1].ssfi_k.mean())

# --------------------------------------------------------------- attribution
att = {}
for cid, rg in [("DR1", 0), ("DR1", 2), ("PU1", 0), ("PU1", 3), ("DR4", 0), ("DR3", 2)]:
    r = mx[(mx.claim == cid) & (mx.regime == rg)].iloc[0]
    att[f"{cid}-{rg}"] = dict(sum=r.shapley_sum, by_cat=r.shapley_by_cat, top=r.shapley_top[:8], vban=r.vbanzhaf_top[:8],
                             split_rank_rho=r.shapley_split_rank_rho, margin_A=r.margin_A, margin_U=r.margin_U)
S["attribution"] = att


# ---------------------------------------------------------------- inference
def cluster_boot(df, stat, B=4000):
    cl = df.claim.unique()
    out = []
    for _ in range(B):
        samp = rng.choice(cl, len(cl), replace=True)
        db = pd.concat([df[df.claim == k] for k in samp])
        v = stat(db)
        if v is not None and np.isfinite(v):
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], len(out)


def signflip_p(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0
    obs = abs(d.mean())
    cnt = 0
    tot = 0
    for signs in itertools.product([1, -1], repeat=n):
        tot += 1
        cnt += abs((d * np.array(signs)).mean()) >= obs - 1e-15
    return cnt / tot, n


def h1stat(df):
    if df.n_entry.nunique() < 2 or df.ssfi.nunique() < 2:
        return None
    return spearmanr(df.n_entry, df.ssfi).correlation


ci, nb = cluster_boot(rpd, h1stat)
S["H1"] = dict(rho=float(spearmanr(rpd.n_entry, rpd.ssfi).correlation), ci=ci, n=int(len(rpd)), n_claims=int(rpd.claim.nunique()),
               share_zero=float((rpd.n_entry == 0).mean()))
sub = rpd[rpd.n_entry.notna()]
ef = rpd[rpd.family.isin(["DR", "PU"]) & (rpd.claim.isin(["DR1", "DR2", "DR5", "PU1", "PU2", "PU3", "PU4"]))]
ci2, _ = cluster_boot(ef, h1stat)
S["H1_errorfree_subset"] = dict(rho=float(spearmanr(ef.n_entry, ef.ssfi).correlation), ci=ci2, n=int(len(ef)))


def fluxstat(df):
    if df.ssfi.nunique() < 2:
        return None
    return spearmanr(df.entry_flux, df.ssfi).correlation


ci3, _ = cluster_boot(rpd, fluxstat)
S["H1_flux_exploratory"] = dict(rho=float(spearmanr(rpd.entry_flux, rpd.ssfi).correlation), ci=ci3)

y = rpd[rpd.type != "control"]


def h2stat(df):
    if not (df.type == "stability").any() or not df.longrun.any():
        return None
    return df[df.type == "stability"].ssfi.mean() - df[df.longrun].ssfi.mean()


ci, nb = cluster_boot(y, h2stat)
S["H2"] = dict(diff=float(h2stat(y)), ci=ci, stab=float(y[y.type == "stability"].ssfi.mean()), longrun=float(y[y.longrun].ssfi.mean()),
               n_stab=int((y.type == "stability").sum()), n_longrun=int(y.longrun.sum()),
               claim_means_stab={c: float(v) for c, v in y[y.type == "stability"].groupby("claim").ssfi.mean().items()},
               claim_means_longrun={c: float(v) for c, v in y[y.longrun].groupby("claim").ssfi.mean().items()})

# H3: pre-specified comparison and claim-level analyses
rows = []
for (c, r), df in C[C.universe == "U_max"].groupby(["claim", "regime"]):
    a = df.set_index("dyn")
    if not a.loc["D0", "verdict_A"]:
        continue
    flips = np.mean([a.loc[d, "verdict_A"] != a.loc["D0", "verdict_A"] for d in ["D1", "D2", "D3", "D4", "D5"]])
    rows.append(dict(claim=c, regime=r, dyn_flip=flips, ssfi_max=a.loc["D0", "ssfi"]))
H3 = pd.DataFrame(rows)
w = wilcoxon(H3.ssfi_max, H3.dyn_flip)
cm = H3.groupby("claim")[["ssfi_max", "dyn_flip"]].mean()
d_claim = (cm.ssfi_max - cm.dyn_flip).values
p_sf, n_nz = signflip_p(d_claim)
ci_h3, _ = cluster_boot(H3.assign(d=H3.ssfi_max - H3.dyn_flip), lambda df: df.d.mean())
S["H3"] = dict(ssfi=float(H3.ssfi_max.mean()), flip=float(H3.dyn_flip.mean()), n=int(len(H3)),
               wilcoxon_rowlevel_p=float(w.pvalue), wilcoxon_W=float(w.statistic),
               more=int((H3.ssfi_max > H3.dyn_flip).sum()), less=int((H3.ssfi_max < H3.dyn_flip).sum()),
               ties=int((H3.ssfi_max == H3.dyn_flip).sum()),
               claim_mean_diff=float(d_claim.mean()), claim_signflip_p=float(p_sf), claim_nonzero=int(n_nz),
               cluster_ci_mean_diff=ci_h3)
H3.to_csv(os.path.join(RES, "h3_pairs.csv"), index=False)

# specification multiverse (SpFI) comparison
spf = []
for f in sorted(glob.glob(os.path.join(RES, "spec_*.json"))):
    if os.path.basename(f).startswith("spec_") and not os.path.basename(f).startswith("spec_primary"):
        try:
            spf.extend(json.load(open(f)))
        except Exception:
            pass
if spf:
    SP = pd.DataFrame([{k: v for k, v in r.items() if k != "draws"} for r in spf])
    SP = SP.merge(H3[["claim", "regime", "ssfi_max"]], on=["claim", "regime"], how="inner")
    SP.to_csv(os.path.join(RES, "spec_multiverse.csv"), index=False)
    cm2 = SP.groupby("claim")[["ssfi_max", "spfi"]].mean()
    d2 = (cm2.ssfi_max - cm2.spfi).values
    p2, n2 = signflip_p(d2)
    ci_sp, _ = cluster_boot(SP.assign(d=SP.ssfi_max - SP.spfi), lambda df: df.d.mean())
    S["SpFI"] = dict(spfi_mean=float(SP.spfi.mean()), ssfi_mean=float(SP.ssfi_max.mean()), n=int(len(SP)),
                     more=int((SP.ssfi_max > SP.spfi).sum()), less=int((SP.ssfi_max < SP.spfi).sum()),
                     ties=int((SP.ssfi_max == SP.spfi).sum()), claim_mean_diff=float(d2.mean()),
                     claim_signflip_p=float(p2), claim_nonzero=int(n2), cluster_ci=ci_sp,
                     by_claim={c: [float(a), float(b)] for c, (a, b) in cm2.iterrows()},
                     fermi_only=float(SP.spfi_fermi_only.mean()), moran_only=float(SP.spfi_moran_only.mean()))
    fm = []
    for r in spf:
        for mu, v in (r.get("finite_mu") or {}).items():
            if isinstance(v, list) and v and v[0] in (True, False):
                fm.append(dict(claim=r["claim"], regime=r["regime"], mu=float(mu), verdict=v[0], margin=v[1], coop=v[2],
                               verdict_D0=r["verdict_D0"], margin_D0=r["margin_D0"]))
    if fm:
        FM = pd.DataFrame(fm)
        FM.to_csv(os.path.join(RES, "finite_mu_authors.csv"), index=False)
        S["finite_mu_authors"] = {str(mu): dict(n=int(len(g)), flips=int((g.verdict != g.verdict_D0).sum()),
                                                flipped=[f"{a}-{b}" for a, b in g[g.verdict != g.verdict_D0][["claim", "regime"]].values])
                                  for mu, g in FM.groupby("mu")}

# exploratory margin predictor
zz = rpd[rpd.type != "stability"]
zz = zz.assign(am=zz.margin_A.abs())
cmz = zz.groupby("claim")[["am", "ssfi"]].mean()
ci_m, _ = cluster_boot(zz, lambda df: spearmanr(df.am, df.ssfi).correlation if df.ssfi.nunique() > 1 else None)
# standardised margin: distance to threshold in units of the spread of superset margins
zz = zz.assign(spread=(zz.q95 - zz.q05) / 3.29)
zz = zz.assign(zstd=zz.am / zz.spread.replace(0, np.nan))
S["margin_predictor"] = dict(rho_rows=float(spearmanr(zz.am, zz.ssfi).correlation), n_rows=int(len(zz)),
                             n_claim_regimes=int(zz.groupby(["claim", "regime"]).ngroups), n_claims=int(zz.claim.nunique()),
                             rho_claim_means=float(spearmanr(cmz.am, cmz.ssfi).correlation), cluster_ci=ci_m,
                             rho_standardised=float(spearmanr(zz.zstd.fillna(np.inf), zz.ssfi).correlation))

# authors' set versus multiverse median (unique model-regime-universe combinations)
SAmap = {c["id"]: ",".join(c["SA"]) for c in CL.CLAIMS}
mm = rpd[(rpd.universe == "U_max") & rpd.q50.notna()].copy()
mm["model_key"] = mm.family + "|" + mm.claim.map(SAmap) + "|" + mm.regime_par.astype(str)
uniq = mm[mm.coop_q50.notna()].drop_duplicates("model_key")
S["authors_vs_median"] = dict(rows=int(len(mm)), unique_models=int(len(uniq)),
                              margin_above_median_rows=int((mm.margin_A > mm.q50).sum()),
                              coop_above_median_unique=int((uniq.coop_A > uniq.coop_q50).sum()),
                              coop_below_median_unique=int((uniq.coop_A < uniq.coop_q50).sum()),
                              controls_unique=int((uniq.type == "control").sum()),
                              controls_coop_below_median=int(((uniq.coop_A < uniq.coop_q50) & (uniq.type == "control")).sum()))

# ------------------------------------------------------- dynamics robustness
dyn_rows = []
for (c, u, dn), g in C[~C.dup].groupby(["claim", "universe", "dyn"]):
    rr = g[g.verdict_A]
    dyn_rows.append(dict(claim=c, universe=u, dyn=dn, reproduced=int(len(rr)), regimes=int(len(g)),
                         ssfi_mean=float(rr.ssfi.mean()) if len(rr) else np.nan,
                         reversed_complete=int((~rr.verdict_U).sum()) if len(rr) else 0,
                         ssfi_q_mean=float(rr.ssfi_q.mean()) if len(rr) and rr.ssfi_q.notna().any() else np.nan))
DY = pd.DataFrame(dyn_rows)
DY.to_csv(os.path.join(RES, "dynamics_table2.csv"), index=False)
dm = C[(C.universe == "U_max")]
okD0 = dm[dm.dyn == "D0"].set_index(["claim", "regime"]).verdict_A.rename("repD0")
dm = dm.join(okD0, on=["claim", "regime"])
S["ssfi_by_dyn"] = {d: dict(mean=float(g.ssfi.mean()), n=int(len(g)), reversed=int((~g.verdict_U).sum()))
                    for d, g in dm[dm.repD0 & dm.verdict_A].groupby("dyn")}
piv = dm.pivot_table(index=["claim", "regime"], columns="dyn", values="verdict_A", aggfunc="first")
piv = piv[piv.D0.astype(bool)]
S["verdict_flip_by_dyn"] = {d: float((piv[d] != piv.D0).mean()) for d in piv.columns if d != "D0"}
ir_d2 = dm[(dm.family == "IR") & (dm.dyn == "D2")][["claim", "regime", "verdict_A", "margin_A", "margin_U", "verdict_U", "ssfi", "ssfi_ci"]]
S["IR_D2"] = ir_d2.to_dict(orient="records")

# ------------------------------------------------------------ predicates
pu2 = rp[rp.claim == "PU2"][["regime", "universe", "coop_A", "ctrl_A", "coop_U", "ctrl_U", "ssfi", "ssfi_rep"]]
S["PU2"] = pu2.to_dict(orient="records")
pu3 = rp[rp.claim == "PU3"][["regime", "universe", "strict_margin_A", "strict_margin_U"]]
S["PU3_strict"] = pu3.to_dict(orient="records")

# threshold sensitivity for level/control claims (D0, all universes, distinct sets)
th_rows = []
for _, r in d0[(~d0.dup) & d0.type.isin(["level", "control"])].iterrows():
    f = os.path.join(RES, f"spec_primary_{r.claim}_{r.regime}_{r.ulevel}.npz")
    if not os.path.exists(f):
        continue
    zf = np.load(f, allow_pickle=True)
    coop = zf["coop"].astype(float)
    for th in (0.4, 0.45, 0.5, 0.55, 0.6):
        sign = 1 if r.type == "level" else -1
        vA = sign * (r.coop_A - th) > 1e-9
        vU = sign * (r.coop_U - th) > 1e-9
        vs = sign * (coop - th) > 1e-9
        th_rows.append(dict(claim=r.claim, regime=r.regime, universe=r.universe, theta=th, verdict_A=bool(vA),
                            verdict_U=bool(vU), ssfi=float(np.mean(vs != vA)) if len(coop) else np.nan))
TH = pd.DataFrame(th_rows)
TH.to_csv(os.path.join(RES, "threshold_sensitivity.csv"), index=False)
S["threshold"] = {str(th): dict(reproduced=int(g[g.universe == "U_max"].verdict_A.sum()),
                                reversed=int((g[g.universe == "U_max"].verdict_A & ~g[g.universe == "U_max"].verdict_U).sum()),
                                ssfi_mean=float(g[(g.universe == "U_max") & g.verdict_A].ssfi.mean()))
                  for th, g in TH.groupby("theta")}

# ------------------------------------------------------------ variants
V = {}
for v in ("ir_smallest_root", "ir_full_selfexcl", "pu_punish_if_game"):
    X = load(f"cells_{v}_s*.jsonl")
    if X.empty:
        continue
    X = X[X.dup_of.isna()]
    base = d0[~d0.dup].set_index(["claim", "regime", "universe"])
    X = X.join(base[["verdict_A", "verdict_U", "ssfi", "margin_U", "coop_U"]], on=["claim", "regime", "universe"], rsuffix="_p")
    V[v] = dict(cells=int(len(X)), verdict_A_changed=int((X.verdict_A != X.verdict_A_p).sum()),
                verdict_U_changed=int((X.verdict_U != X.verdict_U_p).sum()),
                max_abs_coopU_change=float((X.coop_U - X.coop_U_p).abs().max()),
                ssfi_mean=float(X[X.verdict_A].ssfi.mean()), ssfi_mean_primary=float(X[X.verdict_A_p].ssfi_p.mean()),
                changed=[f"{a}-{b}-{c}" for a, b, c in X[(X.verdict_U != X.verdict_U_p) | (X.verdict_A != X.verdict_A_p)][["claim", "regime", "universe"]].values],
                pu1_umax=X[(X.claim == "PU1") & (X.universe == "U_max")][["regime", "coop_U", "coop_U_p", "verdict_U"]].to_dict(orient="records") if v.startswith("pu") else None)
S["variants"] = V
EX = load("cells_primary_expl_s*.jsonl")
if not EX.empty:
    EX = EX[EX.dup_of.isna()]
    e0 = EX[EX.dyn == "D0"]
    S["exploratory"] = e0[["claim", "regime", "regime_par", "universe", "verdict_A", "coop_A", "coop_U", "verdict_U", "ssfi", "coop_Uq", "verdict_Uq"]].to_dict(orient="records")
    EX.to_csv(os.path.join(RES, "exploratory_cells.csv"), index=False)

# ------------------------------------------------------- finite mutation sims
fms = []
for f in sorted(glob.glob(os.path.join(RES, "fmu_*.json"))):
    fms.extend(json.load(open(f)))
if fms:
    S["fmu_sims"] = fms
for nm in ("diagnostics.json", "validation2.json"):
    p = os.path.join(RES, nm)
    if os.path.exists(p):
        S[nm.split(".")[0]] = json.load(open(p))
json.dump(S, open(os.path.join(RES, "summary2.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("ok")
