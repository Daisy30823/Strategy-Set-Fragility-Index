# The garden of forking strategy sets

This research package accompanies revision 1 of *The garden of forking strategy sets: a multiverse audit of evolutionary models of cooperation*. The strategy-set fragility index (SSFI) measures how often a qualitative verdict changes when omitted strategies are added to the authors' strategy set. The audit covers direct reciprocity, optional public goods with punishment, and indirect reciprocity. The frozen corpus comprises 14 claims in 51 parameter regimes.

## Install and run

Use 64-bit CPython 3.11 or 3.12 and run the commands from the package root. The manuscript reports Python 3.11.16. The pinned dependencies were exercised on Windows with CPython 3.12.13. Python, pip and a shell are sufficient for all numerical outputs and figures.

Run these commands in Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py stored
```

Run these commands in a macOS or Linux shell.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py stored
```

The `stored` workflow expands the archived simulation outputs, recalculates the summaries and inference, and builds Table 1, the main and Extended Data figures, and the Supplementary Information LaTeX source. A network connection is needed to install dependencies from PyPI unless they are available locally. Both workflows run offline after installation.

Use the same interpreter with `reproduce.py full --jobs 4` to recompute the simulations from the model definitions and then run the same analysis. Default destinations are `output/stored/` and `output/full/`. Pass `--output path/to/new-directory` to change the destination. A relative value is interpreted from the calling directory. The destination must be new or empty. The runner uses one BLAS and Numba thread per process. See [Simulation protocol](SIMULATIONS.md) for the command sequence, parameters and runtimes.

## Package contents

| Path | Contents |
| --- | --- |
| `reproduce.py`, `requirements.txt` | Workflow entry point and pinned Python dependencies |
| `code/engine.py`, `code/claims.py`, `code/audit2.py` | Payoffs, fixation and stationary solvers, claim corpus, parameters and primary audit |
| `code/spec_multiverse.py`, `code/fmu_sim.py` | Specification multiverse, exact finite-mutation solves and individual-based simulations |
| `code/validate2.py`, `code/diagnostics.py` | Independent numerical checks |
| `code/analyze2.py`, `code/table1.py`, `code/build_si.py` | Statistics and table generation |
| `code/run_figures.py`, `code/make_figs2.py`, `code/make_ed2.py`, `code/figstyle.py` | Figure entry point, panels and styling |
| `analysis/` | Frozen plan, revision amendment and standalone SSFI listing printed in the Supplementary Information |
| `data/archived_runs.zip` | Revision run inputs for the stored workflow |

## Data origin and formats

All results were generated computationally. The claim configurations in `code/claims.py` reconstruct published model regimes. Supplementary Table S1 records the literature provenance. The workflows require no external datasets, credentials, APIs or author-specific directories.

| Archive members | Records |
| --- | --- |
| `cells_primary_s*.jsonl` | 16 shards with 1,224 cells, including 1,128 distinct cells. Fields include `claim`, zero-based `regime`, `regime_par`, `universe`, `ulevel`, `dyn`, `dup_of`, cooperation, margins, verdicts, SSFI, intervals and attribution summaries. |
| `cells_ir_smallest_root_s*.jsonl`, `cells_ir_full_selfexcl_s*.jsonl`, `cells_pu_punish_if_game_s*.jsonl` | Nine shards with 120 D0 convention-variant cells. |
| `cells_primary_expl_s*.jsonl` | Four shards with 480 exploratory cells. D0 is used in the reported exploratory panels. |
| `spec_primary_*.npz` | 59 draw-level files. They contain margins, cooperation, subset sizes, packed masks, labels and attribution arrays. Names encode the claim, regime and universe generator. `masks` uses NumPy packbits with columns ordered by the omitted strategies in `audit2.run_task`. Use `n_omitted` to trim padding. |
| `spec_0.json` through `spec_3.json` | 51 records with 300 individual evolutionary specifications each and exact finite-mutation results where applicable. |
| `fmu_0.json` through `fmu_2.json` | 18 individual-based cases, four replicates per case, with small-mutation reference values and Monte Carlo standard errors. |
| `validation2.json`, `diagnostics.json` | Payoff, fixation, finite-mutation, clone, reputation-root and stationary-solver checks. |

The audit uses `U_enc`, `U_eq` and `U_max` for the encoding, intermediate and largest universes. `dup_of` identifies copied results for identical strategy sets. The original JSON/JSONL serialization can contain the nonstandard token `NaN` for missing values. Use Python's JSON reader or an equivalent reader that accepts this token. NumPy archives use numeric and Unicode arrays.

## Expected outputs

Each output directory contains the raw inputs to the analysis, `summary2.json`, `table1.csv`, eight analysis CSVs, `fig1`–`fig6` and `ed1`–`ed6` in PDF and PNG, and `supplementary.tex` with Tables S1–S20. The Supplementary Information can be compiled separately by running `pdflatex supplementary.tex` twice from the output directory with a TeX installation that supplies the packages named in the preamble. This package was verified through generation of the TeX source.

Useful checkpoints in `summary2.json` are 47 reproduced regimes under D0, 10 complete-largest-universe reversals across five claims, mean labelled SSFI of 0.106600, 0.140921 and 0.179587 in the three universes, and an H3 exact sign-flip probability of 0.015625. See the [results map](RESULTS_MAP.md) for the input and output locations of each figure and table.

The full workflow can take hours on a small machine. Figures may differ in rendering across fonts and Matplotlib builds.
