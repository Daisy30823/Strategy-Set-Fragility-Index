"""Rebuild revision-1 outputs from archived runs or by rerunning the audit."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
ARCHIVE = ROOT / "data" / "archived_runs.zip"
AUDITS = (
    ("primary", "", 16, ()),
    ("ir_smallest_root", "", 3, ("--light",)),
    ("ir_full_selfexcl", "", 3, ("--light",)),
    ("pu_punish_if_game", "", 3, ("--light",)),
    ("primary", "_expl", 4, ("--light", "--exploratory")),
)
CURVES = {
    "DR1_0_m1grid", "DR4_0_m1grid", "PU1_0_rn24", "IR1_0_grid81", "DR1_2_m1grid",
}


def simulation_jobs():
    for variant, _, count, flags in AUDITS:
        for shard in range(count):
            yield ("audit2.py", ".", "1", "--variant", variant, *flags,
                   "--shard", str(shard), "--nshards", str(count))
    for script, stem, count in (("spec_multiverse.py", "spec", 4), ("fmu_sim.py", "fmu", 3)):
        for shard in range(count):
            yield (script, f"{stem}_{shard}.json", str(shard), str(count))
    yield ("validate2.py", "validation2.json")
    yield ("diagnostics.py", "diagnostics.json")


def unpack(output):
    with ZipFile(ARCHIVE) as archive:
        for entry in archive.infolist():
            if not re.fullmatch(r"(cells|spec|fmu)_[A-Za-z0-9_]+\.(jsonl|json|npz)|(?:validation2|diagnostics)\.json", entry.filename):
                raise ValueError(f"Unexpected archive entry: {entry.filename}")
        archive.extractall(output)


def check_inputs(output):
    for variant, suffix, count, _ in AUDITS:
        for shard in range(count):
            path = output / f"cells_{variant}{suffix}_s{shard}.jsonl"
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing or empty audit shard: {path.name}")
    primary = []
    for shard in range(16):
        path = output / f"cells_primary_s{shard}.jsonl"
        with path.open(encoding="utf-8") as handle:
            primary.extend(json.loads(line) for line in handle)
    if len(primary) != 1224:
        raise ValueError(f"Expected 1224 primary cells, found {len(primary)}")

    curves = CURVES | {
        f"{r['claim']}_{r['regime']}_{r['ulevel']}" for r in primary
        if r["dyn"] == "D0" and not r.get("dup_of") and r["type"] in {"level", "control"}
    }
    for curve in sorted(curves):
        path = output / f"spec_primary_{curve}.npz"
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty specification archive: {path.name}")

    spec = []
    for shard in range(4):
        spec.extend(json.loads((output / f"spec_{shard}.json").read_text(encoding="utf-8")))
    if len(spec) != 51 or any(len(row["draws"]) != 300 for row in spec):
        raise ValueError("Expected 51 specification records with 300 draws each")
    for row in spec:
        if any(result[0] == "error" for result in row["finite_mu"].values()):
            raise ValueError(f"Finite-mutation solve failed for {row['claim']} regime {row['regime']}")
    finite = []
    for shard in range(3):
        finite.extend(json.loads((output / f"fmu_{shard}.json").read_text(encoding="utf-8")))
    if len(finite) != 18 or any(len(row["sim"]) != 4 for row in finite):
        raise ValueError("Expected 18 finite-mutation cases with four replicates each")
    for name in ("validation2.json", "diagnostics.json"):
        json.loads((output / name).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stored", "full"))
    parser.add_argument("--output", type=Path, help="New or empty directory (default: output/MODE beside this script)")
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent simulation processes in full mode (default: 2)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = (args.output or ROOT / "output" / args.mode).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="ssfi-") as cache:
        env = os.environ.copy()
        env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(Path(cache) / "mpl"),
                   NUMBA_CACHE_DIR=str(Path(cache) / "numba"), PYTHONDONTWRITEBYTECODE="1")
        for library in ("OMP", "OPENBLAS", "MKL", "NUMBA", "BLIS"):
            env[f"{library}_NUM_THREADS"] = "1"

        def run(job):
            script, *arguments = job
            print("+", script, *arguments, flush=True)
            subprocess.run([sys.executable, str(CODE / script), *arguments],
                           cwd=output, env=env, check=True)

        if args.mode == "stored":
            unpack(output)
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                for _ in pool.map(run, simulation_jobs()):
                    pass
        check_inputs(output)
        for script in ("analyze2.py", "table1.py", "run_figures.py", "build_si.py"):
            run((script, "."))
    print(f"Outputs written to {output}")


if __name__ == "__main__":
    main()
