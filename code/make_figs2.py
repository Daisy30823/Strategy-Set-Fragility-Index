"""Revised main and Extended Data figures. Run inside the python kernel (apply_figure_style, panel_letter in namespace).
Reads rev_out/ outputs only."""
import glob, json, ast
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = "rev_out"
MM = 1 / 25.4
apply_figure_style(sizes=(7, 6, 5.5))
FAM_C = {"DR": "#3B6FB6", "PU": "#C8701E", "IR": "#3E9C6E"}
HOLD, REV, GREY = "#5A5A5A", "#C0392B", "#B8B8B8"
CAT_C = {"entry": "#7B3294", "neutral": "#E6A532", "other": "#9AA5B1"}
UN = ["U_enc", "U_eq", "U_max"]
UN_LAB = {"U_enc": "encoding", "U_eq": "intermediate", "U_max": "largest"}
ORDER = ["DR1", "DR2", "DR3", "DR4", "DR5", "DR6", "PU1", "PU2", "PU3", "PU4", "IR1", "IR2", "IR3", "IR4"]
S = json.load(open(f"{OUT}/summary2.json"))
cells = pd.concat([pd.read_json(p, lines=True) for p in glob.glob(f"{OUT}/cells_primary_s*.jsonl")], ignore_index=True)
d0 = cells[cells.dyn == "D0"].copy()
rep = d0[d0.verdict_A == True]
FIGS = {}


def fam(c):
    return c[:2]


def rlab(c, ri):
    rp = d0[(d0.claim == c) & (d0.regime == ri)].regime_par.iloc[0]
    if c.startswith("DR"):
        return f"{c} $\\delta$={rp['delta']:g}"
    if c.startswith("PU"):
        return f"{c} r={rp['r']:g}, $\\gamma$={rp['cost']:g}"
    return f"{c} {rp['norm']} b={rp['b']:g}"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    fig.savefig(f"{OUT}/{name}.pdf")
    FIGS[name] = fig


# ============================================================ Fig 1 =========
fig = plt.figure(figsize=(180 * MM, 96 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.30, left=0.035, right=0.985, top=0.90, bottom=0.10)
ax = fig.add_subplot(gs[0])
rows = [("DR", "Direct reciprocity: AllC, TFT, AllD", [3, 8, 32, 243], [None, 8, 26, 227], ["comp", "exp", "comp"]),
        ("PU", "Punishment and participation: C, D, L, P", [4, 9, 12, 24], [None, 9, 12, 24], ["comp", "exp", "exp"]),
        ("IR", "Indirect reciprocity: AllC, AllD, DISC", [3, 4, 16, 81], [None, 4, 16, 81], ["comp", "exp", "comp"])]
labs = ["authors' set", "encoding", "intermediate", "largest"]
mk = ["*", "o", "s", "D"]
for k, (f, title, sizes, qs, kinds) in enumerate(rows):
    yy = 2 - k
    for j in range(3):
        ax.plot(sizes[j:j + 2], [yy, yy], color=FAM_C[f], lw=1.1, ls="-" if kinds[j] == "comp" else (0, (2.2, 1.6)), zorder=1)
    for li, sz in enumerate(sizes):
        ax.scatter([sz], [yy], marker=mk[li], s=[46, 22, 22, 22][li], color=FAM_C[f] if li else "white",
                   edgecolor=FAM_C[f], lw=1.0, zorder=3)
        lab = f"{sz}" + (f"\n({qs[li]})" if qs[li] is not None and qs[li] != sz else "")
        ax.text(sz, yy - 0.14, lab, ha="center", va="top", fontsize=5.5, color="#222222")
    ax.text(2.25, yy + 0.22, title, color=FAM_C[f], fontsize=6, fontweight="bold", va="bottom")
    if f == "DR":
        ax.scatter([4], [yy], marker="*", s=46, color="white", edgecolor=FAM_C[f], lw=1.0, zorder=4)
        ax.text(4, yy - 0.14, "4", ha="center", va="top", fontsize=5.5, color="#222222")
ax.set_xscale("log")
ax.set_xlim(2.0, 560)
ax.set_xticks([3, 10, 30, 100, 300]); ax.set_xticklabels(["3", "10", "30", "100", "300"])
ax.set_ylim(-1.45, 2.62)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Number of strategies in the universe (log scale)")
ax.set_title("Three nested strategy universes per claim", loc="left")
h1_ = [Line2D([], [], marker=mk[i], ls="", ms=[6, 3.6, 3.6, 3.4][i], mfc="white" if i == 0 else "#444444", mec="#444444", label=labs[i]) for i in range(4)]
leg1 = ax.legend(handles=h1_, loc="lower left", frameon=False, fontsize=5.5, ncol=4, bbox_to_anchor=(0.0, 0.085), handletextpad=0.2, columnspacing=0.9)
ax.add_artist(leg1)
ax.legend(handles=[Line2D([], [], color="#444444", lw=1.1, label="completion within the stated protocol"),
                   Line2D([], [], color="#444444", lw=1.1, ls=(0, (2.2, 1.6)), label="expansion of the protocol")],
          loc="lower left", frameon=False, fontsize=5.5, ncol=2, bbox_to_anchor=(0.0, 0.0), handletextpad=0.3, columnspacing=0.9)
ax.text(0.99, 0.985, "Brackets: distinct behaviours\nwhen moves are error-free", transform=ax.transAxes, fontsize=5.5, color="#444444", ha="right", va="top")
panel_letter(ax, "a", dx=-0.03)

ax = fig.add_subplot(gs[1])
cols = UN + ["U_max_q"]
M = np.full((len(ORDER), 4), np.nan)
A = [["" for _ in range(4)] for _ in ORDER]
for i, c in enumerate(ORDER):
    for j, u in enumerate(cols):
        uu = "U_max" if u == "U_max_q" else u
        sub = rep[(rep.claim == c) & (rep.universe == uu)]
        if len(sub) == 0:
            continue
        val = pd.Series(np.where(sub.U_q_size == sub.U_size, sub.ssfi, sub.ssfi_q)) if u == "U_max_q" else sub.ssfi
        rv = (~sub.verdict_Uq.astype(bool)).sum() if u == "U_max_q" else (~sub.verdict_U.astype(bool)).sum()
        M[i, j] = val.mean()
        A[i][j] = f"{val.mean():.2f}" + (f"\n{rv}/{len(sub)} rev." if rv else "")
cmap = mpl.colormaps["Greys"].copy()
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(len(ORDER)):
    for j in range(4):
        if A[i][j]:
            ax.text(j, i, A[i][j], ha="center", va="center", fontsize=5.0, color="white" if M[i, j] > 0.55 else "#111111",
                    fontweight="bold" if "rev" in A[i][j] else "normal")
        if "rev" in A[i][j]:
            ax.add_patch(plt.Rectangle((j - 0.48, i - 0.47), 0.96, 0.94, fill=False, ec=REV, lw=1.0))
ax.axvline(2.5, color="white", lw=2.5)
for y in (5.5, 9.5):
    ax.axhline(y, color="white", lw=2)
ax.set_xticks(range(4)); ax.set_xticklabels(["encoding", "intermediate", "largest", "largest,\nclones merged"])
ax.set_yticks(range(len(ORDER)))
nrep = {c: f"{(rep.claim == c).sum() // 3}/{(d0.claim == c).sum() // 3}" for c in ORDER}
ax.set_yticklabels([f"{c}  ({nrep[c]})" for c in ORDER])
for t, c in zip(ax.get_yticklabels(), ORDER):
    t.set_color(FAM_C[fam(c)])
ax.tick_params(length=0)
ax.set_title("Fragility concentrates in a few direct-reciprocity and punishment claims", loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Mean SSFI over reproduced regimes")
cb.set_ticks([0, 0.5, 1])
panel_letter(ax, "b", dx=-0.27)
save(fig, "fig1")

# ============================================================ Fig 2 =========
fig = plt.figure(figsize=(180 * MM, 112 * MM))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], hspace=0.62, wspace=0.62, left=0.07, right=0.985, top=0.88, bottom=0.09)
show = [("DR1", 0, "m1grid", "DR1, $\\delta$=0.8"), ("DR4", 0, "m1grid", "DR4, $\\delta$=0.8"),
        ("PU1", 0, "rn24", "PU1, r=3, $\\gamma$=0.3"), ("IR1", 0, "grid81", "IR1, SJ, b=3")]
for k, (c, ri, ul, ttl) in enumerate(show):
    ax = fig.add_subplot(gs[0, k])
    z = np.load(f"{OUT}/spec_primary_{c}_{ri}_{ul}.npz", allow_pickle=True)
    m = np.sort(z["margins"].astype(float))
    x = np.linspace(0, 1, len(m))
    rv = m < 0
    ax.scatter(x[~rv], m[~rv], color=HOLD, s=0.6, lw=0, rasterized=True)
    ax.scatter(x[rv], m[rv], color=REV, s=0.6, lw=0, rasterized=True)
    ax.axhline(0, color="#222222", lw=0.6)
    ax.scatter([-0.07], [float(z["margin_A"])], marker="*", s=40, color="white", ec="black", lw=0.8, zorder=5, clip_on=False)
    ax.scatter([1.07], [float(z["margin_U"])], marker="D", s=16, color="black", zorder=5, clip_on=False)
    ss = float(d0[(d0.claim == c) & (d0.regime == ri) & (d0.universe == "U_max")].ssfi.iloc[0])
    ax.text(0.97, 0.04, f"SSFI = {ss:.2f}", transform=ax.transAxes, va="bottom", ha="right", fontsize=6)
    ax.set_xlim(-0.14, 1.14)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_title(ttl, loc="left", color=FAM_C[fam(c)])
    ax.set_xlabel("Rank of random supersets")
    if k == 0:
        ax.set_ylabel("Claim margin")
    panel_letter(ax, "abcd"[k], dx=-0.30 if k == 0 else -0.22)
fig.legend(handles=[Line2D([], [], marker="*", ls="", ms=7, mfc="white", mec="black", label="authors' set"),
                    Line2D([], [], marker="D", ls="", ms=3.5, color="black", label="complete universe"),
                    Line2D([], [], marker="o", ls="", ms=3, color=HOLD, label="superset, verdict holds"),
                    Line2D([], [], marker="o", ls="", ms=3, color=REV, label="superset, verdict reversed")],
           loc="center left", bbox_to_anchor=(0.06, 0.975), ncol=4, frameon=False, fontsize=5.5, handletextpad=0.3, columnspacing=1.4)
# e: SSFI(k)
kk = pd.read_csv(f"{OUT}/ssfi_by_k.csv")
ax = fig.add_subplot(gs[1, 0:2])
for (c, ri), g in kk.groupby(["claim", "regime"]):
    g = g.sort_values("k")
    if g.ssfi_k.max() <= 0:
        continue
    lab = f"{c}-{ri}"
    ax.plot(g.k / g.m, g.ssfi_k, color=FAM_C[g.family.iloc[0]], lw=0.9, marker="o", ms=1.8,
            alpha=1.0 if (c, ri) in [("DR1", 0), ("DR4", 0), ("PU1", 0)] else 0.35, zorder=3 if (c, ri) in [("DR1", 0), ("DR4", 0), ("PU1", 0)] else 2)
    OFF = {("DR4", 0): (0.25, (-4, 7), "right"), ("DR1", 0): (0.25, (5, -10), "left"), ("PU1", 0): (0.04, (-4, 3), "right")}
    if (c, ri) in OFF:
        th, off, ha = OFF[(c, ri)]
        gg = g[(g.k / g.m) > th].iloc[0]
        ax.annotate(rlab(c, ri), (gg.k / gg.m, gg.ssfi_k), xytext=off, textcoords="offset points", fontsize=5.5,
                    color=FAM_C[g.family.iloc[0]], ha=ha, fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("Added strategies k as a share of all omitted strategies (log scale)")
ax.set_ylabel("SSFI(k)")
ax.set_ylim(-0.03, 1.03)
ax.set_title("Fragility grows with the number of added strategies", loc="left")
ax.legend(handles=[Line2D([], [], color=FAM_C[f], lw=1, label=l) for f, l in
                   [("DR", "direct reciprocity"), ("PU", "punishment"), ("IR", "indirect reciprocity")]],
          frameon=False, loc="upper left", fontsize=5.5)
panel_letter(ax, "e", dx=-0.13)
# f: SSFI by q
ax = fig.add_subplot(gs[1, 2])
qs = [0.05, 0.1, 0.25, 0.5, 0.75]
um = rep[rep.universe == "U_max"]
for _, r in um.iterrows():
    d = r.ssfi_by_q if isinstance(r.ssfi_by_q, dict) else None
    if not d or max(d.values()) <= 0:
        continue
    ax.plot(qs, [d[str(q)] for q in qs], color=FAM_C[r.family], lw=0.8, alpha=0.6, marker="o", ms=1.5)
mq = S["ssfi_by_q_mean"]
ax.plot(qs, [mq[str(q)] for q in qs], color="black", lw=1.6, marker="s", ms=3, zorder=5)
ax.text(0.5, mq["0.5"] - 0.06, "mean of 47 cells", fontsize=5.5, ha="center", va="top", fontweight="bold")
ax.set_xlabel("Inclusion probability q")
ax.set_ylabel("SSFI(q)")
ax.set_ylim(-0.03, 1.03)
ax.set_xlim(0, 0.8)
ax.set_title("SSFI rises with q", loc="left")
panel_letter(ax, "f", dx=-0.30)
# g: single reversers
ax = fig.add_subplot(gs[1, 3])
sr = S["single_reversers"]
items = sorted([(k, v) for k, v in sr.items() if v > 0], key=lambda t: -t[1])
ax.barh(range(len(items)), [v for _, v in items], color=[FAM_C[k[:2]] for k, _ in items], height=0.7)
ax.set_yticks(range(len(items))); ax.set_yticklabels([rlab(k.split("-")[0], int(k.split("-")[1])) for k, _ in items], fontsize=5)
ax.invert_yaxis()
ax.set_xlabel("Single strategies that reverse")
ax.set_title("Single reversers: 14 cells", loc="left")
panel_letter(ax, "g", dx=-0.62)
save(fig, "fig2")

# ============================================================ Fig 3 =========
qt = pd.read_csv(f"{OUT}/quotient_table.csv")
fig = plt.figure(figsize=(180 * MM, 62 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.0, 1.25], wspace=0.45, left=0.065, right=0.985, top=0.86, bottom=0.2)
ax = fig.add_subplot(gs[0])
ct = S["diagnostics"]["R3_unit"]
vals = [ct["clone_triad"]["labelled_coop"], ct["clone_triad_plus_ALLC_clone"]["labelled_coop"], ct["clone_triad_plus_ALLC_clone"]["quotient_coop"]]
ax.bar(range(3), vals, color=[FAM_C["DR"], REV, "#7FA6D9"], width=0.65)
for i, v_ in enumerate(vals):
    ax.text(i, v_ + 0.015, f"{v_:.2f}", ha="center", va="bottom", fontsize=5.5)
ax.set_xticks(range(3)); ax.set_xticklabels(["triad", "triad +\nAllC copy", "clones\nmerged"])
ax.set_ylim(0, 0.9)
ax.set_ylabel("Cooperation (SML)")
ax.set_title("An AllC copy lowers cooperation", loc="left")
panel_letter(ax, "a", dx=-0.32)
ax = fig.add_subplot(gs[1])
q2 = qt[qt.ssfi.notna() & qt.ssfi_q.notna()]
for f_ in ["DR", "PU", "IR"]:
    g = q2[q2.claim.str[:2] == f_]
    ax.scatter(g.ssfi, g.ssfi_q, s=12, color=FAM_C[f_], lw=0, alpha=0.85)
ax.plot([0, 1], [0, 1], color="#888888", lw=0.6, ls=":")
dr1 = q2[(q2.claim == "DR1") & (q2.regime == 0) & (q2.universe == "U_max")].iloc[0]
ax.annotate("DR1, $\\delta$=0.8", (dr1.ssfi, dr1.ssfi_q), xytext=(0.48, 0.30), fontsize=5.5,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444"))
ax.set_xlabel("SSFI, labelled strategies")
ax.set_ylabel("SSFI, clone classes")
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
ax.text(0.03, 0.97, "error-free direct-reciprocity cells\n(other cells have no clones)", transform=ax.transAxes, va="top", fontsize=5.5)
ax.set_title(f"Merging clones changes {S['quotient_changed_cells']} cells", loc="left")
panel_letter(ax, "b", dx=-0.27)
ax = fig.add_subplot(gs[2])
dr = rep[(rep.claim == "DR1")]
dl = {0: 0.8, 1: 0.85, 2: 0.9, 3: 0.95}
xs = np.arange(4)
w = 0.38
for j, (col, lab_, cc) in enumerate([("coop_U", "labelled", FAM_C["DR"]), ("coop_Uq", "clones merged", "#7FA6D9")]):
    ys = []
    for ri in range(4):
        g = dr[(dr.regime == ri) & (dr.universe == "U_max")]
        ys.append(float(g[col].iloc[0]) if len(g) else np.nan)
    ax.bar(xs + (j - 0.5) * w, ys, width=w, color=cc, label=lab_)
ca = [float(dr[(dr.regime == ri) & (dr.universe == "U_max")].coop_A.iloc[0]) for ri in range(4)]
ax.scatter(xs, ca, marker="*", s=36, color="white", ec="black", lw=0.7, zorder=4, label="authors' triad")
ax.axhline(0.5, color=REV, lw=0.7, ls="--", label="claim threshold")
ax.set_xticks(xs); ax.set_xticklabels([f"{dl[r]}" for r in range(4)])
ax.set_xlabel("Continuation probability $\\delta$ (DR1, 243-strategy universe)")
ax.set_ylabel("Cooperation (SML)")
ax.set_ylim(0, 0.95)
ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=5.5, columnspacing=0.8, handletextpad=0.3)
ax.set_title("The DR1 reversal is carried by duplicated behaviours", loc="left")
panel_letter(ax, "c", dx=-0.17)
save(fig, "fig3")

# ============================================================ Fig 4 =========
fig = plt.figure(figsize=(180 * MM, 74 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.8, 0.9], wspace=0.75, left=0.07, right=0.985, top=0.86, bottom=0.17)
for k, (c, ri, ul, n_top, ttl) in enumerate([("DR1", 0, "m1grid", 14, "DR1, $\\delta$=0.8: neutral GRIM-like rules"),
                                             ("PU1", 0, "rn24", 10, "PU1, r=3, $\\gamma$=0.3: DpL and LpC")]):
    ax = fig.add_subplot(gs[k])
    z = np.load(f"{OUT}/spec_primary_{c}_{ri}_{ul}.npz", allow_pickle=True)
    sh, se, nm, ca_ = z["shapley"], z["shapley_se"], z["omitted"], z["cats"]
    o = np.argsort(-np.abs(sh))[:n_top][::-1]
    ax.barh(range(len(o)), sh[o], xerr=1.96 * se[o], color=[CAT_C[ca_[i]] for i in o], height=0.7,
            error_kw=dict(lw=0.5, capsize=1.2, ecolor="#333333"))
    ax.set_yticks(range(len(o))); ax.set_yticklabels([nm[i] for i in o], fontsize=5.5)
    ax.axvline(0, color="#222222", lw=0.6)
    ax.set_xlabel("Shapley contribution to margin")
    ax.set_title(ttl, loc="left", fontsize=6)
    panel_letter(ax, "ab"[k], dx=-0.3)
    if k == 0:
        ax.legend(handles=[Patch(color=CAT_C[x_], label=l_) for x_, l_ in
                           [("entry", "can enter the authors' set"), ("neutral", "neutral, cannot enter"), ("other", "cannot enter, not neutral")]],
                  frameon=False, loc="lower right", fontsize=5.2, handlelength=1)
ax = fig.add_subplot(gs[2])
att = S["attribution"]
keys = ["DR1-0", "DR1-2", "DR4-0", "DR3-2", "PU1-0", "PU1-3"]
y = np.arange(len(keys))
for j, cat in enumerate(["entry", "neutral", "other"]):
    vals = np.array([att[k_]["by_cat"][cat][1] for k_ in keys])
    ax.barh(y + (j - 1) * 0.26, vals, height=0.26, color=CAT_C[cat])
ax.scatter([att[k_]["sum"] for k_ in keys], y, marker="|", s=60, color="black", zorder=4)
ax.axvline(0, color="#222222", lw=0.6)
ax.set_yticks(y); ax.set_yticklabels([rlab(k_.split("-")[0], int(k_.split("-")[1])) for k_ in keys], fontsize=5.2)
ax.invert_yaxis()
ax.set_xlabel("Summed Shapley contribution")
ax.set_title("Entry and neutral strategies pull apart", loc="left", fontsize=6)
panel_letter(ax, "c", dx=-0.3)
save(fig, "fig4")

# ============================================================ Fig 5 =========
fig = plt.figure(figsize=(180 * MM, 64 * MM))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.8, 1.1, 1], wspace=0.55, left=0.06, right=0.985, top=0.84, bottom=0.2)
ax = fig.add_subplot(gs[0])
h1 = rep[rep.dup_of.isna() & (rep.universe != "U_enc") | (rep.dup_of.isna() & (rep.universe == "U_enc"))]
h1 = rep[rep.dup_of.isna()]
for f_ in ["DR", "PU", "IR"]:
    g = h1[h1.family == f_]
    jit = np.random.default_rng(1).uniform(-0.25, 0.25, len(g))
    ax.scatter(g.n_entry + jit, g.ssfi, s=7, color=FAM_C[f_], lw=0, alpha=0.8)
ax.set_xscale("symlog", linthresh=1)
ax.set_xticks([0, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0", "1", "2", "5", "10", "20"])
ax.set_xlabel("Omitted entry strategies (jittered)")
ax.set_ylabel("SSFI")
h = S["H1"]
ax.set_title(f"H1 not supported: $\\rho$ = {h['rho']:.2f}", loc="left", fontsize=6)
ax.text(0.97, 0.97, f"95% CI {h['ci'][0]:.2f} to {h['ci'][1]:.2f}\n{h['n']} distinct cells, {h['n_claims']} claims", transform=ax.transAxes, va="top", ha="right", fontsize=5.2)
ax.set_ylim(-0.05, 1.08)
panel_letter(ax, "a", dx=-0.32)
ax = fig.add_subplot(gs[1])
h2 = S["H2"]
for j, (key, lab_) in enumerate([("claim_means_stab", "stability"), ("claim_means_longrun", "long-run")]):
    vals = h2[key]
    xs_ = j + np.linspace(-0.15, 0.15, len(vals))
    ax.scatter(xs_, list(vals.values()), s=12, color=[FAM_C[k_[:2]] for k_ in vals], lw=0)
    ax.plot([j - 0.25, j + 0.25], [np.mean(list(vals.values()))] * 2, color="black", lw=1.2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["stability", "long-run"])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("Claim-mean SSFI (largest universe)")
ax.set_title("H2 not supported", loc="left", fontsize=6)
ax.set_ylim(-0.05, 1.05)
panel_letter(ax, "b", dx=-0.42)
ax = fig.add_subplot(gs[2])
h3 = pd.read_csv(f"{OUT}/h3_pairs.csv")
sp = S["SpFI"]["by_claim"]
cm_ = h3.groupby("claim").agg(ssfi=("ssfi_max", "mean"), flip=("dyn_flip", "mean")).reindex([c for c in ORDER if c in set(h3.claim)])
yy = np.arange(len(cm_))
ax.scatter(cm_.ssfi, yy, color="black", s=14, label="SSFI (strategy set)", zorder=3)
ax.scatter(cm_.flip, yy, facecolor="white", edgecolor="black", s=14, label="D1 to D5 flip rate", zorder=3)
ax.scatter([sp[c][1] for c in cm_.index], yy, marker="x", color="#666666", s=14, label="SpFI (300 specifications)", zorder=3)
for i_, c in enumerate(cm_.index):
    ax.plot([min(cm_.flip[c], sp[c][1], cm_.ssfi[c]), max(cm_.flip[c], sp[c][1], cm_.ssfi[c])], [i_, i_], color=GREY, lw=0.6, zorder=1)
ax.set_yticks(yy); ax.set_yticklabels(cm_.index)
for t, c in zip(ax.get_yticklabels(), cm_.index):
    t.set_color(FAM_C[c[:2]])
ax.invert_yaxis()
ax.set_xlabel("Fragility (claim mean)")
H3 = S["H3"]
ax.set_title("H3: set exceeds specifications", loc="left", fontsize=6)
ax.text(0.97, 0.40, f"claim-level sign-flip\nP = {H3['claim_signflip_p']:.3f}", transform=ax.transAxes, ha="right", fontsize=5.2)
ax.legend(frameon=False, loc="lower right", fontsize=5, handletextpad=0.2, bbox_to_anchor=(1.02, 0.0))
panel_letter(ax, "c", dx=-0.25)
ax = fig.add_subplot(gs[3])
mp = rep[(rep.universe == "U_max") & rep.type.isin(["level", "winner", "comparative", "control"])]
for f_ in ["DR", "PU", "IR"]:
    g = mp[mp.family == f_]
    ax.scatter(g.margin_A, g.ssfi, s=10, color=FAM_C[f_], lw=0)
mpS = S["margin_predictor"]
ax.set_xlabel("Margin with the authors' set")
ax.set_ylabel("SSFI (largest universe)")
ax.set_title(f"Larger margins, lower fragility", loc="left", fontsize=6)
ax.text(0.97, 0.97, f"claim-mean $\\rho$ = {mpS['rho_claim_means']:.2f}\n({mpS['n_claims']} claims, {mpS['n_claim_regimes']} regimes)", transform=ax.transAxes,
        ha="right", va="top", fontsize=5.2)
ax.set_ylim(-0.05, 1.05)
panel_letter(ax, "d", dx=-0.3)
save(fig, "fig5")

# ============================================================ Fig 6 =========
var = {vn: pd.concat([pd.read_json(p, lines=True) for p in glob.glob(f"{OUT}/cells_{vn}_s*.jsonl")], ignore_index=True)
       for vn in ["ir_smallest_root", "ir_full_selfexcl", "pu_punish_if_game"]}
DY = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]
DYL = ["D0 Fermi\n$\\beta$=1", "D1 weak\n$\\beta$=0.1", "D2 strong\n$\\beta$=10", "D3 Moran", "D4 N/2", "D5 2N", "D6 local\nmutation", "D7 N=50"]
VL = ["IR smaller\nroot", "IR full self-\nexclusion", "PU punish only\nin game"]
M = np.full((len(ORDER), len(DY) + 3), np.nan)
A = [["" for _ in range(M.shape[1])] for _ in ORDER]
for i, c in enumerate(ORDER):
    for j, dname in enumerate(DY):
        g = cells[(cells.claim == c) & (cells.dyn == dname) & (cells.universe == "U_max") & (cells.verdict_A == True)]
        ntot = ((cells.claim == c) & (cells.dyn == dname) & (cells.universe == "U_max")).sum()
        if len(g):
            M[i, j] = g.ssfi.mean()
            rv = (~g.verdict_U.astype(bool)).sum()
            A[i][j] = f"{g.ssfi.mean():.2f}" + (f"\n{rv}/{len(g)}" if rv else "") + ("" if len(g) == ntot else "*")
        elif ntot:
            A[i][j] = "n.r."
    for j, vn in enumerate(var):
        g = var[vn][(var[vn].claim == c) & (var[vn].universe == "U_max") & (var[vn].verdict_A == True)]
        if len(g):
            M[i, len(DY) + j] = g.ssfi.mean()
            rv = (~g.verdict_U.astype(bool)).sum()
            A[i][len(DY) + j] = f"{g.ssfi.mean():.2f}" + (f"\n{rv}/{len(g)}" if rv else "")
        else:
            A[i][len(DY) + j] = "\u2013"
fig, ax = plt.subplots(figsize=(180 * MM, 100 * MM))
fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.04)
im = ax.imshow(M, cmap=mpl.colormaps["Greys"], vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if A[i][j]:
            ax.text(j, i, A[i][j], ha="center", va="center", fontsize=5.0,
                    color="white" if (not np.isnan(M[i, j]) and M[i, j] > 0.55) else "#111111")
        if "/" in A[i][j]:
            ax.add_patch(plt.Rectangle((j - 0.48, i - 0.47), 0.96, 0.94, fill=False, ec=REV, lw=0.9))
ax.axvline(len(DY) - 0.5, color="white", lw=3)
for y_ in (5.5, 9.5):
    ax.axhline(y_, color="white", lw=2)
ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(DYL + VL, fontsize=5.2)
ax.xaxis.tick_top()
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER)
for t, c in zip(ax.get_yticklabels(), ORDER):
    t.set_color(FAM_C[c[:2]])
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("Mean SSFI, largest universe"); cb.set_ticks([0, 0.5, 1])
fig.text(0.07, 0.965, "Evolutionary specifications (left) and modelling conventions (right): the same claims stay fragile", fontsize=7, ha="left")
fig.text(0.07, 0.01, "Cells: mean SSFI over reproduced regimes; second line, regimes reversed / reproduced (red frame); * not all regimes reproduced; n.r. not reproduced; \u2013 not applicable.",
         fontsize=5.2)
save(fig, "fig6")
print("main figures done")
