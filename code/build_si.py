"""Build supplementary.tex in the supplied results directory."""
import glob
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import claims as CL

OUT = Path(sys.argv[1])
INPUTS = Path(__file__).resolve().parent.parent / "analysis"
S = json.load(open(f"{OUT}/summary2.json"))
cells = pd.concat([pd.read_json(p, lines=True) for p in sorted(glob.glob(f"{OUT}/cells_primary_s*.jsonl"))], ignore_index=True)
d0 = cells[cells.dyn == "D0"].copy()
ORDER = [c["id"] for c in CL.CLAIMS]
UN = ["U_enc", "U_eq", "U_max"]
ULAB = {"U_enc": "enc", "U_eq": "int", "U_max": "L"}
NAME = {"ALLC": "AllC", "ALLD": "AllD", "TFT": "TFT", "WSLS": "WSLS", "DISC": "DISC", "C": "C", "D": "D", "L": "L", "P": "P"}
PLAN = (INPUTS / "analysis_plan_frozen.md").read_text(encoding="utf-8")
AMEND = (INPUTS / "analysis_plan_amendment.md").read_text(encoding="utf-8")
CODEBOX = (INPUTS / "ssfi_minimal.py").read_text(encoding="utf-8")


def esc(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


UMAP = {"×": r"$\times$", "β": r"$\beta$", "δ": r"$\delta$", "λ": r"$\lambda$", "μ": r"$\mu$", "–": "--", "…": r"\dots{}",
        "∈": r"$\in$", "−": "$-$", "≤": r"$\le$", "í": r"\'{\i}"}


def md2tex(md):
    out, inlist = [], False
    for line in md.splitlines():
        t = esc(line.rstrip())
        t = re.sub(r"\^\(([^)]*)\)", lambda m: "$^{" + m.group(1).replace("\u2212", "-") + "}$", t)
        t = re.sub(r"10\^(\u2212?\d+)", lambda m: "$10^{" + m.group(1).replace("\u2212", "-") + "}$", t)
        t = t.replace("<=", "\u2264").replace(">=", "\u2265")
        t = t.replace("|", "$|$").replace("<", "$<$").replace(">", "$>$").replace("\u2265", "$\\ge$")
        t = re.sub(r'"([^"]*)"', r"``\1''", t)
        for k, v in UMAP.items():
            t = t.replace(k, v)
        if t.startswith("\\#"):
            if inlist:
                out.append(r"\end{itemize}"); inlist = False
            lvl = len(line) - len(line.lstrip("#"))
            title = t.lstrip("\\#").strip()
            out.append((r"\paragraph{" if lvl > 1 else r"\subsection*{") + title + "}")
        elif t.startswith("- "):
            if not inlist:
                out.append(r"\begin{itemize}"); inlist = True
            out.append(r"\item " + t[2:])
        else:
            if inlist and t.strip():
                out.append(r"\end{itemize}"); inlist = False
            out.append(t)
    if inlist:
        out.append(r"\end{itemize}")
    return "\n".join(out)


def f3(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def rlab(c, rp):
    if c.startswith("DR"):
        return rf"$\delta$={rp['delta']:g}" + (rf", $\varepsilon$={rp['eps']:g}" if rp["eps"] else "")
    if c.startswith("PU"):
        return rf"r={rp['r']:g}, $\gamma$={rp['cost']:g}"
    return f"{rp['norm']}, b={rp['b']:g}"


def ssfi_str(r):
    if r.verdict_A is not True and r.verdict_A != 1:
        return "n.r."
    if pd.isna(r.ssfi):
        return "--"
    v = float(r.ssfi)
    if not r.ssfi_exact:
        if v == 0:
            return r"$<5\!\times\!10^{-5}$"
        if v >= 1:
            return r"$>0.9999$"
    if v >= 0.9995:
        return r"$>0.999$"
    return f"{v:.3f}"


parts = [r"""\documentclass[10pt]{article}
\usepackage[margin=2.0cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage{listings}
\usepackage[hidelinks]{hyperref}
\AtBeginDocument{\lefthyphenmin=2 \righthyphenmin=3}
\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,frame=single}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thesection}{S\arabic{section}}
\title{Supplementary Information for\\ ``The garden of forking strategy sets: a multiverse audit of evolutionary models of cooperation''}
\author{[Author 1]}
\date{}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""]

# S1 ---------------------------------------------------------------------------
SRC = {"DR1": "Axelrod \\& Hamilton 1981; Nowak et al. 2004; Imhof et al. 2005; Garc\\'{\\i}a \\& Traulsen 2025 (Fig. 1)",
       "DR2": "Axelrod \\& Hamilton 1981 (shadow of the future); negative control",
       "DR3": "Nowak \\& Sigmund 1993; Imhof et al. 2007", "DR4": "Imhof et al. 2007",
       "DR5": "Axelrod \\& Hamilton 1981; Nowak et al. 2004 (finite-population stability of TFT)",
       "DR6": "Nowak \\& Sigmund 1993; Imhof et al. 2007; prior art on instability: Boyd \\& Lorberbaum 1987, Lorberbaum 1994",
       "PU1": "Hauert et al. 2007; Garc\\'{\\i}a \\& Traulsen 2025 (Fig. 2)", "PU2": "Hauert et al. 2002, 2007",
       "PU3": "Hauert et al. 2007", "PU4": "Hauert et al. 2007",
       "IR1": "Ohtsuki \\& Iwasa 2004; Pacheco et al. 2006; Garc\\'{\\i}a \\& Traulsen 2025 (Fig. 3)",
       "IR2": "Nowak \\& Sigmund 1998; Leimar \\& Hammerstein 2001; negative control",
       "IR3": "Ohtsuki \\& Iwasa 2004; Pacheco et al. 2006", "IR4": "Ohtsuki \\& Iwasa 2006; Pacheco et al. 2006"}
PRED = {"level": r"$C(S) > 0.5$", "control": r"$C(S) < 0.5$", "winner": r"$x_X > \max_Y x_Y$ (named rivals)",
        "comparative": r"$C_{\mathrm{mech}} > C_{\mathrm{ctrl}}$", "stability": r"$\max_Y \rho_{Y\to X} \le 1/N$"}
rows = []
for c in CL.CLAIMS:
    regs = "; ".join(rlab(c["id"], r) for r in c["regimes"])
    core = ", ".join(NAME.get(x, x) for x in c.get("core", []))
    rows.append(f"{c['id']} & {esc(c['text'])} & {PRED[c['type']]} & {', '.join(NAME.get(x, x) for x in c['SA'])} & "
                f"{NAME.get(c.get('focal'), c.get('focal') or '--')} & {core or '--'} & {regs} & {SRC[c['id']]} \\\\")
parts.append(r"""\section{Claim corpus, provenance and encodings}
Direct-reciprocity strategies are written $p_0\,p_{CC}\,p_{CD}\,p_{DC}\,p_{DD}$, the probability of cooperating in the first round and after each outcome of the previous round (own action first), with symbols 1, 0 and h ($=\tfrac12$). Named strategies are AllC (11111), AllD (00000), TFT (11010), STFT (01010), WSLS (11001) and GRIM (11000). Punishment strategies are written as the first-stage action (C, D or L) followed by ``p'' and the first-stage actions of the targets they punish; P denotes CpD, a contributor who punishes defectors. Indirect-reciprocity rules are written $p_{GG}\,p_{GB}\,p_{BG}\,p_{BB}$, the cooperation probability given the donor's own and the recipient's reputation; AllC is 1111, AllD 0000 and DISC 1010. Table~\ref{tab:s1} lists every claim with its predicate, authors' set, focal strategy, core strategies (kept in replacement sets), regimes and the literature it represents. The claims are reconstructions of model classes; the sources indicate where the corresponding statement is made or used, not that the source reported exactly our regimes or population sizes. The stability predicate is the finite-population, small-mutation property that no single mutant of another strategy in the set fixes with more than neutral probability; it is not the classical ESS condition. The PU3 winner predicate compares punishers with the authors' rival strategies C, D and L; the strict reading against every other strategy in the set is reported in Section~S8.

{\scriptsize
\begin{longtable}{p{0.55cm}p{3.0cm}p{1.6cm}p{1.6cm}p{0.8cm}p{1.1cm}p{3.0cm}p{3.4cm}}
\caption{Frozen claim corpus with provenance.}\label{tab:s1}\\
\toprule
ID & Claim & Predicate & $S_A$ & Focal & Core & Regimes & Represented literature\\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}}
""")

# S2 ---------------------------------------------------------------------------
def coopA(c, ri):
    return float(d0[(d0.claim == c) & (d0.regime == ri)].coop_A.iloc[0])


def coopU(c, ri, u):
    return float(d0[(d0.claim == c) & (d0.regime == ri) & (d0.universe == u)].coop_U.iloc[0])


anch = [("Repeated PD, $\\delta=0.9$, $N=20$: AllC, TFT, AllD", coopA("DR1", 2), "reactive (8)", coopU("DR1", 2, "U_enc"),
         "memory-one (32)", coopU("DR1", 2, "U_eq"), "grid (243)", coopU("DR1", 2, "U_max")),
        ("Punishment, $r=3$, $\\gamma=0.3$, $N=100$: C, D, L, P", coopA("PU1", 0), "9 strategies", coopU("PU1", 0, "U_enc"),
         "12 strategies", coopU("PU1", 0, "U_eq"), "24 strategies", coopU("PU1", 0, "U_max")),
        ("Stern judging, $b=3$, $N=50$: AllC, AllD, DISC", coopA("IR1", 0), "4 rules", coopU("IR1", 0, "U_enc"),
         "16 rules", coopU("IR1", 0, "U_eq"), "81 rules", coopU("IR1", 0, "U_max"))]
parts.append(r"""\section{Anchor configurations}
Table~\ref{tab:s2} gives long-run cooperation in the small-mutation limit for the anchor configurations used to verify the implementation against ref.~[Garc\'{\i}a \& Traulsen 2025]. The first two examples of that paper are reproduced. The third example of that paper concerns endogenous norms under private assessment and lies outside the public-reputation model used here; the stern-judging row is our own anchor.

\begin{table}[h]\centering\small
\caption{Long-run cooperation for the anchor configurations (primary specification D0).}\label{tab:s2}
\begin{tabular}{p{4.6cm}cp{1.6cm}cp{1.7cm}cp{1.6cm}c}
\toprule
Model and authors' set & $S_A$ & Universe 1 & & Universe 2 & & Universe 3 & \\
\midrule
""" + "\n".join(f"{a} & {b:.3f} & {c} & {d:.3f} & {e} & {f:.3f} & {g} & {h:.3f}\\\\" for a, b, c, d, e, f, g, h in anch) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# S3 ---------------------------------------------------------------------------
V = S["validation2"]
v1 = pd.DataFrame(V["V1_DR"])
v1p = pd.DataFrame(V["V1_PU"])
irv = pd.DataFrame(V["V1_IR"])
v2 = pd.DataFrame(V["V2"])
v3 = pd.DataFrame(V["V3"])
r2 = pd.DataFrame(S["diagnostics"]["R2_solver"])
t3a = "\n".join(f"DR & {esc(r.i)} vs {esc(r.k)} & {r.exact:.4f} & {r.mc:.4f} & {r.se:.4f} & {r.z:+.2f}\\\\" for r in v1.itertuples()) + "\n" + \
      "\n".join(f"PU & {esc(r.i)} vs {esc(r.k)} ($j={r.j}$) & {r.exact:.4f} & {r.mc:.4f} & {r.se:.4f} & {r.z:+.2f}\\\\" for r in v1p.itertuples())
t3b = "\n".join(f"{r.norm} & {'+'.join(r.rules)} ({r.counts[0]}/{r.counts[1]}) & {r.g_meanfield[0]:.4f}, {r.g_meanfield[1]:.4f} & "
                f"{r.g_selfexcl[0]:.4f}, {r.g_selfexcl[1]:.4f} & {r.g_abm[0]:.4f}, {r.g_abm[1]:.4f} & {r.pay_meanfield[0]:.4f}, {r.pay_meanfield[1]:.4f} & "
                f"{r.pay_selfexcl[0]:.4f}, {r.pay_selfexcl[1]:.4f} & {r.pay_abm[0]:.4f}, {r.pay_abm[1]:.4f}\\\\" for r in irv.itertuples())
t3c = "\n".join(f"{r.mutant} into {r.resident} & {r.exact:.4g} & {r.mc:.4g} & {r.se:.2g}\\\\" for r in v2.itertuples())
v3["err"] = [np.abs(np.array(a) - np.array(b)).max() for a, b in zip(v3.sml, v3.finite)]
t3d = "\n".join(f"{esc(r.model)} & {r.mu:g} & {r.coop_sml:.4f} & {r.coop_finite_approx:.4f} & {r.err:.4f}\\\\" for r in v3.itertuples())
parts.append(r"""\section{Validation}
Payoffs, reputations, fixation probabilities and long-run abundances were validated against independent simulations (Tables~\ref{tab:s3}--\ref{tab:s6}; Methods). Indirect-reciprocity reputations and payoffs are compared for the production convention (recipients' reputations through the population mean including the focal individual, payoffs excluding self-interaction) and for full self-exclusion; the agent-based runs record $2\times10^{6}$ interactions after a burn-in of $2\times10^{5}$ and exclude self-interaction throughout. Fixation probabilities were validated by simulation for direct reciprocity only; for indirect reciprocity, reputations and payoffs were validated and the uniqueness of the reputation equilibrium was verified exhaustively (Section~S11).

{\small
\begin{longtable}{llrrrr}
\caption{Exact against Monte Carlo payoffs (DR: 40,000 games per pair, $\delta=0.9$, $\varepsilon=0.01$; PU: 20,000 group draws, $N=30$, $j$ focal players).}\label{tab:s3}\\
\toprule
Family & Pair & Exact & Monte Carlo & SE & $z$\\
\midrule
\endhead
""" + t3a + r"""
\bottomrule
\end{longtable}}

{\scriptsize
\begin{longtable}{llllllll}
\caption{Indirect reciprocity: reputations $g$ and payoffs of the two strategies of each pair (production convention, full self-exclusion, agent-based).}\label{tab:s4}\\
\toprule
Norm & Pair (counts) & $g$ production & $g$ self-excl. & $g$ ABM & payoff production & payoff self-excl. & payoff ABM\\
\midrule
\endhead
""" + t3b + r"""
\bottomrule
\end{longtable}}

\begin{table}[h]\centering\small
\caption{Fixation probabilities: closed form against 20,000 simulated trajectories (Fermi, $\beta=1$, $N=20$, $\delta=0.9$).}\label{tab:s5}
\begin{tabular}{lrrr}
\toprule
Mutant into resident & Exact & Simulated & SE\\
\midrule
""" + t3c + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]\centering\small
\caption{Small-mutation limit against the exact finite-mutation stationary distribution; last column, largest absolute abundance error.}\label{tab:s6}
\begin{tabular}{lrrrr}
\toprule
Model & $\mu$ & $C$, SML & $C$, finite $\mu$ & Max abundance error\\
\midrule
""" + t3d + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# S4 cell table --------------------------------------------------------------------
crow = []
for c in ORDER:
    g = d0[d0.claim == c]
    for ri in sorted(g.regime.unique()):
        for u in UN:
            r = g[(g.regime == ri) & (g.universe == u)].iloc[0]
            if not pd.isna(r.dup_of):
                crow.append(f"{c} & {rlab(c, r.regime_par)} & {ULAB[u]} & \\multicolumn{{9}}{{l}}{{identical to the encoding universe (same 32 strategies)}}\\\\")
                continue
            ci = r.ssfi_ci if isinstance(r.ssfi_ci, list) else [np.nan, np.nan]
            cis = "--" if (r.ssfi_exact or pd.isna(ci[0]) or not r.verdict_A) else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            qv = r.ssfi if r.U_q_size == r.U_size else r.ssfi_q
            crow.append(f"{c} & {rlab(c, r.regime_par)} & {ULAB[u]} & {int(r.U_size)} ({int(r.U_q_size)}) & {int(r.n_omitted)} & "
                        f"{'E' if r.ssfi_exact else 'S'} & {f3(r.margin_A)} & {f3(r.margin_U)} & {'yes' if r.verdict_U else 'no'} & "
                        f"{ssfi_str(r)} & {cis} & {('n.r.' if not r.verdict_A else ('--' if pd.isna(qv) else (r'$>0.999$' if qv >= 0.9995 else f'{qv:.3f}')))} & "
                        f"{'--' if pd.isna(r.ssfi_rep) else f'{r.ssfi_rep:.3f}'}\\\\")
parts.append(r"""\section{Cell-level results under the primary specification}
Table~\ref{tab:s7} lists every claim--regime--universe cell under D0. $|U|$ gives the number of strategies and, in brackets, of behaviourally distinct classes; $m$ is the number of omitted strategies; E marks exact enumeration ($m\le16$ or closed form for stability claims) and S sampling (20,000 supersets) with a 95\% Clopper--Pearson interval. Margins $\Delta(S_A)$ and $\Delta(U)$ are defined in Methods; $V(U)$ is the complete-universe verdict; SSFI$_q$ is the quotient index (equal to SSFI where the universe contains no clones) and Repl. the replacement index. Sampled values of 0 or 1 are shown as bounds. n.r., not reproduced with the authors' set.

{\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{longtable}{lllllllllllll}
\caption{Cell-level results (D0).}\label{tab:s7}\\
\toprule
Claim & Regime & $U$ & $|U|$ & $m$ & E/S & $\Delta(S_A)$ & $\Delta(U)$ & $V(U)$ & SSFI & 95\% CI & SSFI$_q$ & Repl.\\
\midrule
\endhead
""" + "\n".join(crow) + r"""
\bottomrule
\end{longtable}}
""")

# S5 dynamics -------------------------------------------------------------------------
dt = pd.read_csv(f"{OUT}/dynamics_table2.csv")
DY = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]
drow = []
for c in ORDER:
    vals = []
    for dn in DY:
        g = dt[(dt.claim == c) & (dt.universe == "U_max") & (dt.dyn == dn)]
        if len(g) == 0 or g.reproduced.iloc[0] == 0:
            vals.append("n.r.")
        else:
            r = g.iloc[0]
            vals.append(f"{r.ssfi_mean:.2f} ({int(r.reversed_complete)}/{int(r.reproduced)})")
    drow.append(c + " & " + " & ".join(vals) + "\\\\")
H3 = S["H3"]; SP = S["SpFI"]
parts.append(r"""\section{Evolutionary specifications}
Table~\ref{tab:s8} gives the mean largest-universe SSFI over reproduced regimes for every claim and specification, with reversed/reproduced regimes in brackets. D0: Fermi, $\beta=1$, base $N$; D1: $\beta=0.1$; D2: $\beta=10$; D3: Moran process with linear fitness ($w=1$); D4: $N/2$; D5: $2N$; D6: local mutation kernel; D7: $N=50$ in every family.

{\small
\begin{longtable}{lllllllll}
\caption{Largest-universe SSFI by evolutionary specification.}\label{tab:s8}\\
\toprule
Claim & D0 & D1 & D2 & D3 & D4 & D5 & D6 & D7\\
\midrule
\endhead
""" + "\n".join(drow) + r"""
\bottomrule
\end{longtable}}

\paragraph{H3 at the row level.} For continuity with the frozen plan, the pre-specified Wilcoxon signed-rank test across the 47 reproduced claim--regimes gave $W = """ + f"{H3['wilcoxon_W']:.0f}" + r"""$ ($""" + f"{H3['more']}" + r"""$ claim--regimes with larger SSFI, $""" + f"{H3['less']}" + r"""$ with a larger D1--D5 flip rate, $""" + f"{H3['ties']}" + r"""$ ties), $P = """ + f"{H3['wilcoxon_rowlevel_p']:.5f}" + r"""$. This test treats regimes of the same claim as independent and overstates the evidence; the claim-level sign-flip test ($P = """ + f"{H3['claim_signflip_p']:.4f}" + r"""$, """ + f"{H3['claim_nonzero']}" + r""" claims with a non-zero difference) is the primary analysis. Against the specification multiverse, SSFI exceeded SpFI in """ + f"{SP['more']}" + r""" claim--regimes and fell below it in """ + f"{SP['less']}" + r""" (claim-level $P = """ + f"{SP['claim_signflip_p']:.4f}" + r"""$); SpFI averaged """ + f"{SP['fermi_only']:.3f}" + r""" over pairwise-comparison draws and """ + f"{SP['moran_only']:.3f}" + r""" over Moran draws.
""")

# S6 plan + amendment -------------------------------------------------------------------
parts.append(r"""\section{Analysis plan and amendment log}
The frozen analysis plan is reproduced verbatim below, followed by the amendment log of the revision. Changes of numbers relative to the first version are listed at the end of the log.

\subsection*{S6.1 Frozen analysis plan}
{\small """ + md2tex(PLAN) + r"""}

\subsection*{S6.2 Amendment log}
{\small """ + md2tex(AMEND) + r"""}
""")

# S7 attribution ------------------------------------------------------------------------
def att_table(key, ul, label, cap):
    c, ri = key.split("-")
    z = np.load(f"{OUT}/spec_primary_{c}_{ri}_{ul}.npz", allow_pickle=True)
    sh, se, nm, ca = z["shapley"], z["shapley_se"], z["omitted"], z["cats"]
    vb, vbs = z["banzhaf_verdict"], z["banzhaf_verdict_se"]
    o = np.argsort(-np.abs(sh))[:15]
    rr = "\n".join(f"{esc(nm[i])} & {ca[i]} & {sh[i]:+.4f} & {se[i]:.4f} & {vb[i]:+.3f} & {vbs[i]:.3f}\\\\" for i in o)
    a = S["attribution"][key]
    cats = "; ".join(f"{k} ({v[0]} strategies) {v[1]:+.3f}" for k, v in a["by_cat"].items())
    return (r"""\begin{table}[h]\centering\small
\caption{""" + cap + f" Total change in margin {a['sum']:+.3f}; by category: {cats}. Split-half rank correlation {a['split_rank_rho']:.2f}." + r"""}\label{""" + label + r"""}
\begin{tabular}{llrrrr}
\toprule
Strategy & Category & Shapley & SE & Verdict-Banzhaf & SE\\
\midrule
""" + rr + r"""
\bottomrule
\end{tabular}
\end{table}
""")


parts.append(r"""\section{Attribution}
Tables~\ref{tab:s9}--\ref{tab:s12} list the fifteen omitted strategies with the largest absolute Shapley contributions (1,000 orderings, D0, largest universe) together with Monte Carlo standard errors and the verdict-Banzhaf index (change in the probability of reversal when the strategy is included). Categories: entry (neutral with the focal strategy and opening a path for an invader that it resists), neutral (neutral, no such path) and other. Error-prone direct-reciprocity cells contain no neutral strategies. When almost every extension reverses the verdict (for example DR6), including any single strategy changes the reversal probability little, so verdict-Banzhaf values saturate near zero; the single-reverser counts in Fig.~2g complement them.
""" + att_table("DR1-0", "m1grid", "tab:s9", r"DR1, $\delta=0.8$, 243-strategy universe.")
    + att_table("DR1-2", "m1grid", "tab:s10", r"DR1, $\delta=0.9$, 243-strategy universe.")
    + att_table("PU1-0", "rn24", "tab:s11", r"PU1, $r=3$, $\gamma=0.3$, 24-strategy universe.")
    + att_table("DR4-0", "m1grid", "tab:s12", r"DR4, $\delta=0.8$, $\varepsilon=0.01$, 243-strategy universe.") + r"\clearpage" + "\n")

# S8 predicate and threshold ----------------------------------------------------------------
th = S["threshold"]
trow = "\n".join(f"{float(t):.2f} & {th[t]['reproduced']} & {th[t]['reversed']} & {th[t]['ssfi_mean']:.3f}\\\\" for t in sorted(th, key=float))
p3 = pd.DataFrame(S["PU3_strict"])
p3row = "\n".join(f"{rlab('PU3', CL.CLAIMS[8]['regimes'][int(r.regime)])} & {ULAB[r.universe]} & {r.strict_margin_A:+.4f} & {r.strict_margin_U:+.4f}\\\\" for r in p3.itertuples())
pu2 = pd.DataFrame(S["PU2"])
pu2row = "\n".join(f"{rlab('PU2', CL.CLAIMS[7]['regimes'][int(r.regime)])} & {ULAB[r.universe]} & {r.coop_A:.3f} & {r.coop_U:.3f} & {r.ssfi:.3f} & {r.ssfi_rep:.3f}\\\\" for r in pu2.sort_values(['regime', 'universe']).itertuples())
parts.append(r"""\section{Threshold and predicate sensitivity}
\begin{table}[h]\centering\small
\caption{Level claims (21 claim--regimes) as the cooperation threshold $\theta$ varies; largest universe, D0.}\label{tab:s13}
\begin{tabular}{rrrr}
\toprule
$\theta$ & Reproduced & Reversed & Mean SSFI\\
\midrule
""" + trow + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]\centering\small
\caption{PU3 under the strict predicate: abundance of punishers minus the largest abundance of any other strategy in the set, for the authors' set and the complete universe.}\label{tab:s14}
\begin{tabular}{llrr}
\toprule
Regime & $U$ & Strict margin, $S_A$ & Strict margin, $U$\\
\midrule
""" + p3row + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]\centering\small
\caption{PU2: cooperation in the optional arm with the authors' set and the complete universe, superset SSFI and replacement SSFI. Adding strategies cannot remove the loner option, whereas replacement sets of size four that keep only the core strategies usually do; regime $r=3$, $\gamma=1$ has no admissible replacement that reverses.}\label{tab:s15}
\begin{tabular}{llrrrr}
\toprule
Regime & $U$ & $C(S_A)$ & $C(U)$ & SSFI & Replacement SSFI\\
\midrule
""" + pu2row + r"""
\bottomrule
\end{tabular}
\end{table}
\clearpage
""")

# S9 variants and exploratory ---------------------------------------------------------------
vr = S["variants"]
VN = {"ir_smallest_root": "IR: smaller reputation equilibrium", "ir_full_selfexcl": "IR: full self-exclusion",
      "pu_punish_if_game": "PU: punishment only if the game takes place"}
vrow = "\n".join(f"{VN[k]} & {v['cells']} & {v['verdict_A_changed']} & {v['verdict_U_changed']} & {v['max_abs_coopU_change']:.4f} & {v['ssfi_mean_primary']:.4f} & {v['ssfi_mean']:.4f}\\\\" for k, v in vr.items())
ex = pd.DataFrame(S["exploratory"])
EXN = {"DR3x": "DR3", "DR4x": "DR4", "DR6x": "DR6", "IR1x": "IR1", "IR3x": "IR3", "IR4x": "IR4", "IR5x": "SS level", "IR1s": "IR1"}
erow = "\n".join(f"{EXN[r.claim]} & {rlab(r.claim, r.regime_par) if not r.claim.startswith('IR') else r.regime_par['norm'] + ', b=' + format(r.regime_par['b'], 'g')} & {ULAB[r.universe]} & "
                 f"{'yes' if r.verdict_A else 'no'} & {f3(r.coop_A)} & {f3(r.coop_U)} & {'yes' if r.verdict_U else 'no'} & {r.ssfi:.3f}\\\\"
                 for r in ex.itertuples())
parts.append(r"""\section{Modelling variants and exploratory regimes}
Table~\ref{tab:s16} compares the three modelling variants with the primary model over all cells of the affected family (D0). Table~\ref{tab:s17} lists the exploratory regimes, which are excluded from every corpus summary. For the simple-standing rows the verdict is the level predicate $C(S) > 0.5$, which the authors' set does not reproduce; SSFI is then the share of supersets in which the verdict holds.

\begin{table}[h]\centering\small
\caption{Modelling variants: cells evaluated, changes in the authors'-set and complete-universe verdicts, largest absolute change in complete-universe cooperation, and mean SSFI (primary, variant).}\label{tab:s16}
\begin{tabular}{p{5.2cm}rrrrrr}
\toprule
Variant & Cells & $\Delta V(S_A)$ & $\Delta V(U)$ & Max $|\Delta C(U)|$ & SSFI primary & SSFI variant\\
\midrule
""" + vrow + r"""
\bottomrule
\end{tabular}
\end{table}

{\small
\begin{longtable}{llllrrlr}
\caption{Exploratory regimes (D0).}\label{tab:s17}\\
\toprule
Claim & Regime & $U$ & $V(S_A)$ & $C(S_A)$ & $C(U)$ & $V(U)$ & SSFI\\
\midrule
\endhead
""" + erow + r"""
\bottomrule
\end{longtable}}
""")

# S10 finite mutation ------------------------------------------------------------------------
fa = pd.read_csv(f"{OUT}/finite_mu_authors.csv")
farow = "\n".join(f"{r.claim} & {r.regime} & {r.mu:g} & {r.margin_D0:+.4f} & {r.margin:+.4f} & {'yes' if r.verdict else 'no'}\\\\" for r in fa.itertuples())
fm = pd.DataFrame(S["fmu_sims"])
DLT = "$\\delta$="
fm["case_l"] = [esc(c).replace("d=", DLT) for c in fm.case]
fm["U_l"] = [u.replace("m1det", "32").replace("m1grid", "243") for u in fm.universe]
fmrow = "\n".join(f"{r.case_l} & {r.U_l} & {r.mu:g} & {r.sml:.4f} & {r.sim_mean:.4f} & {r.sim_se:.4f} & {r.steps / 1e6:.0f}M\\\\" for r in fm.itertuples())
parts.append(r"""\section{Finite mutation}
\begin{longtable}{lrrrrl}
\caption{Authors'-set verdicts under finite mutation (exact stationary distribution over all compositions); regime index as in Table~\ref{tab:s1} (0-based).}\label{tab:s18}\\
\toprule
Claim & Regime & $\mu$ & $\Delta$, SML & $\Delta$, finite $\mu$ & Verdict\\
\midrule
\endhead
""" + farow + r"""
\bottomrule
\end{longtable}

{\small
\begin{longtable}{llrrrrr}
\caption{Individual-based simulations of DR1 and DR4 in the 32- and 243-strategy universes ($N = 20$, $\beta = 1$, four runs each).}\label{tab:s19}\\
\toprule
Case & $|U|$ & $\mu$ & $C$, SML & $C$, simulation & SE & Steps\\
\midrule
\endhead
""" + fmrow + r"""
\bottomrule
\end{longtable}}
""")

# S11 diagnostics ------------------------------------------------------------------------------
dg = S["diagnostics"]
u = dg["R3_unit"]
rs = "\n".join(f"{r.b:g} & {r.coop_gth:.6g} & {r.coop_hp:.6g} & {r.coop_lu:.3g} & {r.max_abs_err_gth:.1e} & {r.max_abs_err_lu:.3g}\\\\" for r in r2.itertuples())
parts.append(r"""\section{Numerical diagnostics}
Root multiplicity of the reputation map: across all """ + f"{dg['R1_root_multiplicity']['SJ']['compositions']:,}" + r""" two-strategy compositions of the 81-rule universe, the map $F(\bar g)$ had a single root under stern judging and under image scoring (maximum number of roots """ + f"{dg['R1_root_multiplicity']['SJ']['max_roots']}" + r"""). Unit checks: neutral fixation probabilities equal $1/N$ to within """ + f"{max(u['neutral_fix_maxdev_F'], u['neutral_fix_maxdev_M']):.1e}" + r"""; stationary residuals are below """ + f"{max(u['stationary_residual_F'], u['stationary_residual_M']):.1e}" + r"""; the clone toy model gives cooperation """ + f"{u['clone_triad']['labelled_coop']:.4f}" + r""" for the triad, """ + f"{u['clone_triad_plus_ALLC_clone']['labelled_coop']:.4f}" + r""" after adding a relabelled AllC, and """ + f"{u['clone_triad_plus_ALLC_clone']['quotient_coop']:.4f}" + r""" after merging.

\begin{table}[h]\centering\small
\caption{Stationary solver accuracy for the 81-rule stern-judging universe under D2: long-run cooperation by GTH elimination, 120-digit arithmetic and the LU solve of the first version, and largest absolute errors of the stationary distribution.}\label{tab:s20}
\begin{tabular}{rrrrrr}
\toprule
$b$ & $C$, GTH & $C$, 120 digits & $C$, LU & Max error GTH & Max error LU\\
\midrule
""" + rs + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# S12 code box ------------------------------------------------------------------------------------
parts.append(r"""\section{Minimal implementation of the index}
The listing below computes the superset SSFI and the fixed-size index for a level claim from a matrix of pairwise fixation probabilities. It is self-contained; supplied with the engine's fixation matrix for DR1 at $\delta = 0.8$ in the 32-strategy universe it returns 0.380 from 4,000 draws, against 0.382 from 20,000 draws in the audit; the full implementation is in the code package.

\begin{lstlisting}[language=Python]
""" + CODEBOX + r"""
\end{lstlisting}

\section{Software, runtime and reproduction}
All computations used Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, Numba 0.67.0 and mpmath 1.3.0 on a 20-core Windows workstation. The primary audit (1,224 cells) took about 30 minutes with 16 worker processes, the specification multiverse about 25 minutes with 4 processes, and the individual-based finite-mutation simulations about 25 minutes with 3 processes. The README of the code package lists the commands that regenerate every number, table and figure.
\end{document}
""")

(OUT / "supplementary.tex").write_text("\n".join(parts), encoding="utf-8")
print("ok")
