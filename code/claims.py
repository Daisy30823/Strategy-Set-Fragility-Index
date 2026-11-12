"""Frozen corpus of canonical claims (unchanged from the version saved with analysis_plan.md),
plus the revision-stage specifications, which are marked as such.

Each claim: family, type, authors' strategy set S_A (by name), the three nested
universes, the predicate, the focal cooperative strategy (for neutral-entry
counts and stability claims), the core strategies kept in replacement sets,
and the parameter regimes.  Claims attribute a qualitative conclusion to a
model class as commonly studied; parameters are our reconstruction of the
published regime, not a copy of any single paper's figure.
"""

# Strategy names -> family-specific tuples ----------------------------------
DR_S = {"ALLC": (1, 1, 1, 1, 1), "ALLD": (0, 0, 0, 0, 0), "TFT": (1, 1, 0, 1, 0), "WSLS": (1, 1, 0, 0, 1)}
PU_S = {"C": ("C", 0, 0, 0), "D": ("D", 0, 0, 0), "L": ("L", 0, 0, 0), "P": ("C", 0, 1, 0)}
IR_S = {"ALLC": (1, 1, 1, 1), "ALLD": (0, 0, 0, 0), "DISC": (1, 0, 1, 0)}

DR_U = {"triad": ("reactive", "m1det", "m1grid"), "m1": ("m1det", "m1det", "m1grid")}
PU_U = ("gt9", "lone12", "rn24")
IR_U = ("recip4", "own16", "grid81")

DR_BASE = dict(RSTP=(3, 0, 4, 1), N=20)
PU_BASE = dict(n=5, c=1.0, sigma=1.0, N=100)
IR_BASE = dict(c=1.0, eps=0.02, e=0.02, N=50)

DELTAS_HI = (0.8, 0.85, 0.9, 0.95)
PU_REG = ({"r": 3.0, "cost": 0.3, "fine": 1.0}, {"r": 3.0, "cost": 1.0, "fine": 1.0},
          {"r": 2.5, "cost": 0.3, "fine": 1.0}, {"r": 4.0, "cost": 0.3, "fine": 1.0})
IR_B = (3.0, 4.0, 5.0)

CLAIMS = [
    dict(id="DR1", family="DR", type="level", text="Direct reciprocity sustains majority cooperation in the AllC-TFT-AllD model",
         SA=["ALLC", "TFT", "ALLD"], U=DR_U["triad"], focal="TFT", core=["TFT", "ALLD"], theta=0.5, sign=+1,
         regimes=[dict(delta=d, eps=0.0) for d in DELTAS_HI]),
    dict(id="DR2", family="DR", type="control", text="Cooperation fails when interactions are short (AllC-TFT-AllD)",
         SA=["ALLC", "TFT", "ALLD"], U=DR_U["triad"], focal="TFT", core=["TFT", "ALLD"], theta=0.5, sign=-1,
         regimes=[dict(delta=d, eps=0.0) for d in (0.2, 0.25, 0.3)]),
    dict(id="DR3", family="DR", type="winner", text="With execution errors WSLS is more abundant than TFT",
         SA=["ALLC", "ALLD", "TFT", "WSLS"], U=DR_U["m1"], focal="WSLS", core=["WSLS", "TFT", "ALLD"], X="WSLS", rivals=["TFT"],
         regimes=[dict(delta=d, eps=0.01) for d in DELTAS_HI]),
    dict(id="DR4", family="DR", type="level", text="With execution errors the AllC-AllD-TFT-WSLS model sustains majority cooperation",
         SA=["ALLC", "ALLD", "TFT", "WSLS"], U=DR_U["m1"], focal="WSLS", core=["WSLS", "ALLD"], theta=0.5, sign=+1,
         regimes=[dict(delta=d, eps=0.01) for d in DELTAS_HI]),
    dict(id="DR5", family="DR", type="stability", text="TFT cannot be invaded in the AllC-TFT-AllD model",
         SA=["ALLC", "TFT", "ALLD"], U=DR_U["triad"], focal="TFT", core=["TFT", "ALLD"],
         regimes=[dict(delta=d, eps=0.0) for d in DELTAS_HI]),
    dict(id="DR6", family="DR", type="stability", text="With execution errors WSLS cannot be invaded in the AllC-AllD-TFT-WSLS model",
         SA=["ALLC", "ALLD", "TFT", "WSLS"], U=DR_U["m1"], focal="WSLS", core=["WSLS", "ALLD"],
         regimes=[dict(delta=d, eps=0.01) for d in DELTAS_HI]),
    dict(id="PU1", family="PU", type="level", text="Optional participation lets punishment sustain majority cooperation",
         SA=["C", "D", "L", "P"], U=PU_U, focal="P", core=["P", "D"], theta=0.5, sign=+1, regimes=list(PU_REG)),
    dict(id="PU2", family="PU", type="comparative", text="Cooperation is higher with optional than with compulsory participation",
         SA=["C", "D", "L", "P"], U=PU_U, focal="P", core=["P", "D", "L"], arms="drop_loners", regimes=list(PU_REG)),
    dict(id="PU3", family="PU", type="winner", text="Punishers are the most abundant strategy",
         SA=["C", "D", "L", "P"], U=PU_U, focal="P", core=["C", "D", "L", "P"], X="P", rivals=["C", "D", "L"], regimes=list(PU_REG)),
    dict(id="PU4", family="PU", type="stability", text="Punishers cannot be invaded",
         SA=["C", "D", "L", "P"], U=PU_U, focal="P", core=["P", "D"], regimes=list(PU_REG)),
    dict(id="IR1", family="IR", type="level", text="Stern judging sustains majority cooperation among AllC, AllD and DISC",
         SA=["ALLC", "ALLD", "DISC"], U=IR_U, focal="DISC", core=["DISC", "ALLD"], theta=0.5, sign=+1,
         regimes=[dict(norm="SJ", b=b) for b in IR_B]),
    dict(id="IR2", family="IR", type="control", text="Image scoring fails to sustain majority cooperation among AllC, AllD and DISC",
         SA=["ALLC", "ALLD", "DISC"], U=IR_U, focal="DISC", core=["DISC", "ALLD"], theta=0.5, sign=-1,
         regimes=[dict(norm="IS", b=b) for b in IR_B]),
    dict(id="IR3", family="IR", type="comparative", text="Stern judging yields more cooperation than image scoring",
         SA=["ALLC", "ALLD", "DISC"], U=IR_U, focal="DISC", core=["DISC", "ALLD"], arms="norm_SJ_vs_IS",
         regimes=[dict(norm="SJ", b=b) for b in IR_B]),
    dict(id="IR4", family="IR", type="stability", text="DISC cannot be invaded under stern judging",
         SA=["ALLC", "ALLD", "DISC"], U=IR_U, focal="DISC", core=["DISC", "ALLD"],
         regimes=[dict(norm="SJ", b=b) for b in IR_B]),
]

UNIV_TAGS = ("U_enc", "U_eq", "U_max")

# Evolutionary specifications.  D0-D5 are the pre-specified primary and
# alternative specifications.  D6 and D7 were added at revision.
#   name -> dict(dyn=..., mult=population-size multiplier or None, N=absolute N or None,
#                kernel='uniform'|'local')
DYNAMICS = {
    "D0": dict(dyn={"kind": "fermi", "beta": 1.0}, mult=1.0, N=None, kernel="uniform"),
    "D1": dict(dyn={"kind": "fermi", "beta": 0.1}, mult=1.0, N=None, kernel="uniform"),
    "D2": dict(dyn={"kind": "fermi", "beta": 10.0}, mult=1.0, N=None, kernel="uniform"),
    "D3": dict(dyn={"kind": "moran", "w": 1.0}, mult=1.0, N=None, kernel="uniform"),
    "D4": dict(dyn={"kind": "fermi", "beta": 1.0}, mult=0.5, N=None, kernel="uniform"),
    "D5": dict(dyn={"kind": "fermi", "beta": 1.0}, mult=2.0, N=None, kernel="uniform"),
    "D6": dict(dyn={"kind": "fermi", "beta": 1.0}, mult=1.0, N=None, kernel="local"),   # revision: local mutation kernel
    "D7": dict(dyn={"kind": "fermi", "beta": 1.0}, mult=None, N=50, kernel="uniform"),  # revision: common N = 50
}
PRESPECIFIED_DYN = ("D0", "D1", "D2", "D3", "D4", "D5")

# Revision-stage exploratory claim variants (not part of the frozen corpus).
import copy as _copy


def _variant(cid, newid, regimes, note):
    c = _copy.deepcopy(next(x for x in CLAIMS if x["id"] == cid))
    c["id"] = newid
    c["regimes"] = regimes
    c["note"] = note
    c["exploratory"] = True
    return c


EXPLORATORY = [
    _variant("IR1", "IR1x", [dict(norm="SJ", b=b) for b in (2.2, 2.4, 2.6, 2.8)], "margin-matched b/c below the screened range"),
    _variant("IR3", "IR3x", [dict(norm="SJ", b=b) for b in (2.2, 2.4, 2.6, 2.8)], "margin-matched b/c below the screened range"),
    _variant("IR4", "IR4x", [dict(norm="SJ", b=b) for b in (2.2, 2.4, 2.6, 2.8)], "margin-matched b/c below the screened range"),
    _variant("DR3", "DR3x", [dict(delta=0.99, eps=0.01)], "near-undiscounted limit"),
    _variant("DR4", "DR4x", [dict(delta=0.99, eps=0.01)], "near-undiscounted limit"),
    _variant("DR6", "DR6x", [dict(delta=0.99, eps=0.01)], "near-undiscounted limit"),
]
# simple standing (excluded before freezing), documented as a development case
_ss = _copy.deepcopy(next(x for x in CLAIMS if x["id"] == "IR1"))
_ss.update(id="IR5x", text="Simple standing sustains majority cooperation among AllC, AllD and DISC",
           regimes=[dict(norm="SS", b=b) for b in (2.0, 3.0, 4.0, 5.0)], note="excluded before freezing", exploratory=True)
EXPLORATORY.append(_ss)
_sj2 = _copy.deepcopy(next(x for x in CLAIMS if x["id"] == "IR1"))
_sj2.update(id="IR1s", regimes=[dict(norm="SJ", b=2.0)], note="stern judging at b/c = 2 (excluded before freezing)", exploratory=True)
EXPLORATORY.append(_sj2)
