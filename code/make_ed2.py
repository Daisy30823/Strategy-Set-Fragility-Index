"""Extended Data figures (run after make_figs2.py in the same namespace)."""
V = S["validation2"]

# ============================================================ ED1: validation
fig = plt.figure(figsize=(180 * MM, 66 * MM))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.15, 1.05], wspace=0.55, left=0.06, right=0.985, top=0.85, bottom=0.2)
ax = fig.add_subplot(gs[0])
for j, (k, f_) in enumerate([("V1_DR", "DR"), ("V1_PU", "PU")]):
    z_ = pd.DataFrame(V[k]).z.values
    ax.scatter(j + np.linspace(-0.15, 0.15, len(z_)), z_, s=10, color=FAM_C[f_], lw=0)
ax.axhspan(-1.96, 1.96, color="#EEEEEE", zorder=0)
ax.set_xticks([0, 1]); ax.set_xticklabels(["direct\nreciprocity", "punishment"])
ax.set_xlim(-0.5, 1.5); ax.set_ylim(-3, 3)
ax.set_ylabel("z = (simulated - exact) / SE")
ax.set_title("Payoffs vs exact", loc="left")
panel_letter(ax, "a", dx=-0.42)
ax = fig.add_subplot(gs[1])
v2 = pd.DataFrame(V["V2"])
ax.errorbar(v2.exact, v2.mc, yerr=1.96 * v2.se, fmt="o", ms=3, color=FAM_C["DR"], lw=0.6)
lim = [1e-8, 1]
ax.plot(lim, lim, color="#888888", lw=0.6, ls=":")
ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-3)
ax.set_xlabel("Exact fixation probability")
ax.set_ylabel("Simulated fixation probability")
ax.set_title("Fixation (Fermi, N=20)", loc="left")
panel_letter(ax, "b", dx=-0.42)
ax = fig.add_subplot(gs[2])
v3 = pd.DataFrame(V["V3"])
for mdl, cc in zip(v3.model.unique(), [FAM_C["DR"], FAM_C["IR"], FAM_C["PU"]]):
    g = v3[v3.model == mdl].sort_values("mu")
    ax.plot(g.mu, g.coop_finite_approx, marker="o", ms=3, color=cc, lw=1, label=mdl)
    ax.axhline(g.coop_sml.iloc[0], color=cc, lw=0.6, ls="--")
ax.set_xscale("log")
ax.set_xlabel("Mutation rate $\\mu$")
ax.set_ylabel("Cooperation")
ax.legend(frameon=False, fontsize=5.2, loc="center left")
ax.set_title("Finite $\\mu$ vs SML", loc="left")
panel_letter(ax, "c", dx=-0.3)
ax = fig.add_subplot(gs[3])
irv = pd.DataFrame(V["V1_IR"])
lab_ = [f"{r.norm}: {'+'.join(r.rules)}" for _, r in irv.iterrows()]
e_h = [np.abs(np.array(r.pay_meanfield) - np.array(r.pay_abm)).max() for _, r in irv.iterrows()]
e_f = [np.abs(np.array(r.pay_selfexcl) - np.array(r.pay_abm)).max() for _, r in irv.iterrows()]
yy = np.arange(len(irv))
ax.barh(yy - 0.18, e_h, height=0.34, color=FAM_C["IR"], label="production (hybrid)")
ax.barh(yy + 0.18, e_f, height=0.34, color="#9FD3B8", label="full self-exclusion")
ax.set_yticks(yy); ax.set_yticklabels(lab_, fontsize=5)
ax.invert_yaxis()
ax.set_xlabel("Max |payoff - agent-based|")
ax.legend(frameon=False, fontsize=5.2, loc="center right")
ax.set_title("IR payoffs vs agent-based", loc="left")
panel_letter(ax, "d", dx=-0.62)
save(fig, "ed1")

# ============================================================ ED2: replacement and median
fig = plt.figure(figsize=(180 * MM, 64 * MM))
gs = fig.add_gridspec(1, 3, wspace=0.5, left=0.065, right=0.985, top=0.85, bottom=0.2)
ax = fig.add_subplot(gs[0])
pu2 = pd.DataFrame(S["PU2"])
g = pu2[pu2.regime == 0].set_index("universe").reindex(UN)
x_ = np.arange(3)
ax.bar(x_ - 0.2, g.ssfi, width=0.4, color=FAM_C["PU"], label="supersets of the authors' set")
ax.bar(x_ + 0.2, g.ssfi_rep, width=0.4, color="#E8B98A", label="equal-size replacement sets")
for i_, vv in enumerate(g.ssfi):
    ax.plot([i_ - 0.38, i_ - 0.02], [0, 0], color=FAM_C["PU"], lw=1.5)
    ax.text(i_ - 0.2, 0.015, f"{vv:.0f}", ha="center", va="bottom", fontsize=5.5, color=FAM_C["PU"])
ax.set_xticks(x_); ax.set_xticklabels([UN_LAB[u] for u in UN])
ax.set_ylabel("Share of sets reversing PU2")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=5.2, loc="upper left")
ax.set_title("PU2 (r=3, $\\gamma$=0.3): replacing vs adding", loc="left")
panel_letter(ax, "a", dx=-0.3)
ax = fig.add_subplot(gs[1])
mm_ = d0[(d0.universe == "U_max") & d0.coop_q50.notna() & (d0.verdict_A == True)].copy()
SAmap = {c["id"]: ",".join(c["SA"]) for c in CL.CLAIMS} if "CL" in globals() else None
mm_["key"] = mm_.family + "|" + mm_.claim + "|" + mm_.regime_par.astype(str)
hh = []
for t_, mk_ in [("level", "o"), ("winner", "s"), ("comparative", "^"), ("control", "v")]:
    gg = mm_[(mm_.type == t_) & mm_.coop_A.notna()]
    if len(gg) == 0:
        continue
    ax.scatter(gg.coop_q50, gg.coop_A, marker=mk_, s=12, c=[FAM_C[f_] for f_ in gg.family], lw=0)
    hh.append(Line2D([], [], marker=mk_, ls="", ms=3.5, color="#555555", label=t_))
ax.plot([0, 1], [0, 1], color="#888888", lw=0.6, ls=":")
ax.set_xlabel("Median cooperation over random supersets")
ax.set_ylabel("Cooperation with the authors' set")
ax.legend(handles=hh, frameon=False, fontsize=5.2, loc="lower right", title="claim type", title_fontsize=5.2)
ax.set_title("Cooperation: authors' set vs median", loc="left")
panel_letter(ax, "b", dx=-0.3)
ax = fig.add_subplot(gs[2])
mmp = d0[(d0.universe == "U_max") & d0.q50.notna() & (d0.verdict_A == True)]
ax.scatter(mmp.q50, mmp.margin_A, s=10, c=[FAM_C[f_] for f_ in mmp.family], lw=0)
lo, hi = min(mmp.q50.min(), mmp.margin_A.min()), max(mmp.q50.max(), mmp.margin_A.max())
ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.6, ls=":")
ax.set_xlabel("Median margin over random supersets")
ax.set_ylabel("Margin with the authors' set")
avm = S["authors_vs_median"]
ax.set_title(f"Margins: {avm['margin_above_median_rows']} of {avm['rows']} above median", loc="left")
panel_letter(ax, "c", dx=-0.3)
save(fig, "ed2")

# ============================================================ ED3: finite-mutation simulations
fm = pd.DataFrame(S["fmu_sims"])
fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 58 * MM), sharey=True)
fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.2, wspace=0.08)
for ax, case in zip(axs, fm.case.unique()):
    for ul, cc in [("m1det", "#7FA6D9"), ("m1grid", FAM_C["DR"])]:
        g = fm[(fm.case == case) & (fm.universe == ul)].sort_values("mu")
        ax.errorbar(g.mu, g.sim_mean, yerr=1.96 * g.sim_se, fmt="o-", ms=3, lw=0.9, color=cc, capsize=1.5,
                    label=f"{'32 deterministic' if ul == 'm1det' else '243 grid'}: simulation")
        ax.axhline(g.sml.iloc[0], color=cc, ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Mutation rate $\\mu$")
    ax.set_title(case.replace("d=", "$\\delta$="), loc="left")
    ax.axhline(0.5, color=REV, lw=0.5, ls=":")
axs[0].set_ylabel("Cooperation (individual-based, N=20)")
axs[0].legend(frameon=False, fontsize=5.2, loc="lower left")
fig.text(0.07, 0.94, "Dashed lines: small-mutation limit. Error bars: 95% intervals over 4 independent runs.", fontsize=5.5)
for i_, ax in enumerate(axs):
    panel_letter(ax, "abc"[i_], dx=-0.05 if i_ else -0.25)
save(fig, "ed3")

# ============================================================ ED4: threshold and predicate sensitivity
fig = plt.figure(figsize=(180 * MM, 60 * MM))
gs = fig.add_gridspec(1, 2, wspace=0.35, left=0.07, right=0.985, top=0.85, bottom=0.2)
ax = fig.add_subplot(gs[0])
th = S["threshold"]
ts = sorted(th, key=float)
ax.plot([float(t) for t in ts], [th[t]["ssfi_mean"] for t in ts], "o-", color="black", ms=3, label="mean SSFI")
ax2 = ax.twinx()
ax2.bar([float(t) for t in ts], [th[t]["reversed"] for t in ts], width=0.02, color=GREY, alpha=0.8, zorder=0)
ax2.set_ylabel("Reversed level claim-regimes (bars)")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
ax.set_xlabel("Cooperation threshold $\\theta$ for level claims")
ax.set_ylabel("Mean SSFI (largest universe)")
ax.set_title(f"Pre-specified $\\theta$=0.5 lies mid-range", loc="left")
panel_letter(ax, "a", dx=-0.13)
ax = fig.add_subplot(gs[1])
p3 = pd.DataFrame(S["PU3_strict"])
for ri, g in p3.groupby("regime"):
    g = g.set_index("universe").reindex(UN)
    ax.plot(range(3), g.strict_margin_U, marker="o", ms=3, lw=1, color=FAM_C["PU"], alpha=0.4 + 0.15 * ri, label=rlab("PU3", ri)[4:])
ax.axhline(0, color="#222222", lw=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels([UN_LAB[u] for u in UN])
ax.set_ylabel("Strict margin: P minus best rival")
ax.legend(frameon=False, fontsize=5.2, loc="upper right")
ax.set_title("PU3 under the strict 'beats every rival' reading", loc="left")
panel_letter(ax, "b", dx=-0.13)
save(fig, "ed4")

# ============================================================ ED5: exploratory parameters
ex = pd.DataFrame(S["exploratory"])
fig = plt.figure(figsize=(180 * MM, 60 * MM))
gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.065, right=0.985, top=0.85, bottom=0.2)
ax = fig.add_subplot(gs[0])
sj = ex[ex.claim.isin(["IR1x", "IR1s"])].copy()
sj["b"] = sj.regime_par.apply(lambda d: d["b"])
pr = rep[(rep.claim == "IR1")].copy(); pr["b"] = pr.regime_par.apply(lambda d: d["b"])
for u, mk_ in [("U_enc", "o"), ("U_max", "D")]:
    g = pd.concat([sj[sj.universe == u][["b", "coop_U"]], pr[pr.universe == u][["b", "coop_U"]]]).sort_values("b")
    ax.plot(g.b, g.coop_U, marker=mk_, ms=3, lw=1, color=FAM_C["IR"] if u == "U_max" else "#9FD3B8", label=f"{UN_LAB[u]} universe")
ga = pd.concat([sj.drop_duplicates("b")[["b", "coop_A"]], pr.drop_duplicates("b")[["b", "coop_A"]]]).sort_values("b")
ax.plot(ga.b, ga.coop_A, marker="*", ms=5, lw=1, color="black", label="authors' set")
ax.axhline(0.5, color=REV, lw=0.6, ls="--")
ax.axvspan(1.9, 2.9, color="#F2F2F2", zorder=0)
ax.set_xlabel("Benefit b (Stern Judging)")
ax.set_ylabel("Cooperation")
ax.legend(frameon=False, fontsize=5.2, loc="lower right")
ax.set_title("IR1 below b=3 (exploratory, shaded)", loc="left")
panel_letter(ax, "a", dx=-0.3)
ax = fig.add_subplot(gs[1])
ss = ex[ex.claim == "IR5x"].copy(); ss["b"] = ss.regime_par.apply(lambda d: d["b"])
for u, cc in [("U_enc", "#9FD3B8"), ("U_eq", "#5FB38A"), ("U_max", FAM_C["IR"])]:
    g = ss[ss.universe == u].sort_values("b")
    ax.plot(g.b, g.coop_U, marker="o", ms=3, lw=1, color=cc, label=f"{UN_LAB[u]} universe")
g = ss.drop_duplicates("b").sort_values("b")
ax.plot(g.b, g.coop_A, marker="*", ms=5, lw=1, color="black", label="authors' set")
ax.axhline(0.5, color=REV, lw=0.6, ls="--")
ax.set_xlabel("Benefit b (Simple Standing)")
ax.set_ylabel("Cooperation")
ax.legend(frameon=False, fontsize=5.2, loc="upper left")
ax.set_title("Simple Standing (excluded pre-audit)", loc="left")
panel_letter(ax, "b", dx=-0.3)
ax = fig.add_subplot(gs[2])
dx_ = ex[ex.claim.isin(["DR3x", "DR4x", "DR6x"])]
lbl = {"DR3x": "DR3\nWSLS\nwins", "DR4x": "DR4\ncoop.\n> 0.5", "DR6x": "DR6\nWSLS\nstable"}
for i_, c in enumerate(["DR3x", "DR4x", "DR6x"]):
    for j_, u in enumerate(["U_enc", "U_max"]):
        g = dx_[(dx_.claim == c) & (dx_.universe == u)]
        if len(g):
            vv = float(g.ssfi.iloc[0]); rvv = not bool(g.verdict_U.iloc[0])
            ax.text(i_ + (j_ - 0.5) * 0.38, vv + 0.015, f"{vv:.2f}", ha="center", va="bottom", fontsize=5)
            ax.bar(i_ + (j_ - 0.5) * 0.38, vv, width=0.38, color=FAM_C["DR"] if u == "U_max" else "#7FA6D9",
                   edgecolor=REV if rvv else "none", lw=1.0, label=f"{UN_LAB[u]} universe" if i_ == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels([lbl[c] for c in ["DR3x", "DR4x", "DR6x"]])
ax.set_ylabel("SSFI")
ax.set_ylim(0, 1.12)
ax.legend(frameon=False, fontsize=5.2, loc="upper left")
ax.set_title("$\\delta$=0.99, $\\varepsilon$=0.01 (red frame: reversed)", loc="left")
panel_letter(ax, "c", dx=-0.3)
save(fig, "ed5")

# ============================================================ ED6: numerical diagnostics
fig = plt.figure(figsize=(180 * MM, 58 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.4, left=0.07, right=0.985, top=0.85, bottom=0.2)
ax = fig.add_subplot(gs[0])
r2 = pd.DataFrame(S["diagnostics"]["R2_solver"])
xx = np.arange(len(r2))
ax.bar(xx - 0.2, r2.max_abs_err_lu, width=0.4, color=GREY, label="LU solve")
ax.bar(xx + 0.2, np.maximum(r2.max_abs_err_gth, 1e-17), width=0.4, color=FAM_C["IR"], label="GTH elimination")
ax.set_yscale("log")
ax.set_xticks(xx); ax.set_xticklabels([f"b={b:g}" for b in r2.b])
ax.set_ylabel("Max error vs 120-digit solution")
ax.set_xlabel("SJ, 81 rules, strong selection ($\\beta$=10)")
ax.legend(frameon=False, fontsize=5.2, loc="upper right")
ax.set_title("Stationary solver accuracy", loc="left")
panel_letter(ax, "a", dx=-0.25)
ax = fig.add_subplot(gs[1])
ird = pd.DataFrame(S["IR_D2"])
ird["lab"] = [rlab(c, r) for c, r in zip(ird.claim, ird.regime)]
ird = ird.sort_values(["claim", "regime"])
yy = np.arange(len(ird))
ax.barh(yy, ird.ssfi, color=[REV if not v_ else FAM_C["IR"] for v_ in ird.verdict_U], height=0.65)
ax.set_yticks(yy); ax.set_yticklabels(ird.lab, fontsize=5)
ax.invert_yaxis()
ax.set_xlabel("SSFI under strong selection (D2), largest universe")
ax.set_xlim(0, 1)
ax.set_title("Indirect reciprocity under D2 (red: verdict reversed)", loc="left")
panel_letter(ax, "b", dx=-0.3)
save(fig, "ed6")
print("ED done")
