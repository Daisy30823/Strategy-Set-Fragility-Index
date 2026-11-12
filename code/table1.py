"""Export the claim-level source table used for manuscript Table 1."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

from claims import CLAIMS

UNIVERSES = ("U_enc", "U_eq", "U_max")
FIELDS = ("claim", "type", "reg", "rep", "enc", "int_", "L", "Lq", "rev", "revq", "sizes")


def mean_ssfi(rows, quotient=False):
    values = [r["ssfi_q"] if quotient and r["U_q_size"] != r["U_size"] else r["ssfi"] for r in rows]
    return float(np.mean(values)) if values else ""


def export(results):
    cells = []
    for shard in sorted(results.glob("cells_primary_s*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            cells.extend(r for r in map(json.loads, handle) if r["dyn"] == "D0")
    with (results / "table1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELDS)
        for claim in CLAIMS:
            groups = [[r for r in cells if r["claim"] == claim["id"] and r["universe"] == u]
                      for u in UNIVERSES]
            enc, mid, largest = [[r for r in group if r["verdict_A"]] for group in groups]
            sizes = "/".join(str(group[0]["U_size"]) for group in groups)
            writer.writerow((claim["id"], claim["type"], len(groups[2]), len(largest),
                             mean_ssfi(enc), mean_ssfi(mid), mean_ssfi(largest),
                             mean_ssfi(largest, quotient=True),
                             sum(not r["verdict_U"] for r in largest),
                             sum(not r["verdict_Uq"] for r in largest), sizes))


if __name__ == "__main__":
    export(Path(sys.argv[1]))
