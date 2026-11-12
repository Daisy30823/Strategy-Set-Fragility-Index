# Manuscript results and reproduction steps

Run `python reproduce.py stored` to build all outputs below from the archived runs, or `python reproduce.py full` to first regenerate the simulations. Filenames in the tables are relative to the run output directory unless they begin with `code/` or `analysis/`, which are relative to the package root. Primary cells are read from `cells_primary_s*.jsonl`. `summary2.json` keys identify the machine-readable values.

## Main text and Extended Data

| Item | Analysis and inputs | Generated result |
| --- | --- | --- |
| Table 1 and Fig. 1 | `code/table1.py` and `code/make_figs2.py:46`. Primary D0 cells grouped by claim and universe with reproduced regimes as denominator. Fig. 1a shows a schematic with the published strategy-universe sizes. | `table1.csv`, `fig1.pdf`, `fig1.png` |
| Corpus headline results | `code/analyze2.py:31` and `code/analyze2.py:41`. Primary cells yield 1,224 evaluations, 1,128 distinct cells, 47 reproduced regimes, and 10 largest-universe reversals in five claims. Family means and the labelled and quotient statistics are in `field`. Fig. 1 and Table 1 reuse labelled SSFI for unchanged quotient sets. | `summary2.json` (`counts`, `field`), `table1.csv` |
| Fig. 2a–d, specification curves | `code/make_figs2.py:125`. Sorted margins and the authors' and complete-set endpoints from `spec_primary_DR1_0_m1grid.npz`, `spec_primary_DR4_0_m1grid.npz`, `spec_primary_PU1_0_rn24.npz`, and `spec_primary_IR1_0_grid81.npz`. Each has 20,000 margins. | `fig2.pdf`, `fig2.png` |
| Fig. 2e–g, size, inclusion probability and single reversers | `code/analyze2.py:68` and `code/make_figs2.py:155`. Fixed-size and inclusion-probability indices and single-reverser fields from reproduced D0 cells. | `ssfi_by_k.csv`, `summary2.json` (`ssfi_by_q_mean`, `single_reversers`), `fig2.pdf` and PNG |
| Fig. 3, clone quotient | `code/diagnostics.py` (R3), `code/analyze2.py:62` and `code/make_figs2.py:210`. Clone toy example, D0 labelled and quotient comparisons, DR1 at regime 0. | `quotient_table.csv`, `summary2.json` (`diagnostics`, `quotient_changed_cells`), `fig3.pdf` and PNG |
| Fig. 4, attribution | `code/make_figs2.py:263` reads Shapley and Monte Carlo SE arrays for DR1 regime 0 and PU1 regime 0. Panel c uses selected cell-level category totals in `summary2.json` from `code/analyze2.py:83`. | `fig4.pdf`, `fig4.png` |
| Fig. 5a–b, H1 and H2 | `code/analyze2.py:92` and `code/make_figs2.py:299`. Distinct reproduced D0 cells and claim-cluster inference. H2 uses all universes in the supplied plotting data. | `summary2.json` (`H1`, `H1_errorfree_subset`, `H1_flux_exploratory`, `H2`), `fig5.pdf` and PNG |
| Fig. 5c–d, H3, SpFI and exploratory margin | `code/analyze2.py:159` through `code/analyze2.py:227`, `code/spec_multiverse.py:180` (`run`), and `code/make_figs2.py:331`. Primary cells, D1–D5 flips, `spec_*.json` and D0 margins. | `h3_pairs.csv`, `spec_multiverse.csv`, `summary2.json` (`H3`, `SpFI`, `margin_predictor`), `fig5.pdf` and PNG |
| Fig. 6, robustness | `code/analyze2.py:241`, `code/analyze2.py:290` and `code/make_figs2.py:366`. Largest-universe primary D0–D7 cells and D0 convention variants. | `dynamics_table2.csv`, `summary2.json` (`ssfi_by_dyn`, `variants`), `fig6.pdf` and PNG |
| Extended Data Fig. 1, validation | `code/validate2.py` (V1–V3), then `code/make_ed2.py:4` from `validation2.json`. | `ed1.pdf` and PNG |
| Extended Data Fig. 2, replacement and median | `code/analyze2.py:229`, `code/analyze2.py:262` and `code/make_ed2.py:55`. PU2 replacement/superset SSFI and D0 authors'-set outcomes versus multiverse medians. The numeric cooperation summary deduplicates model–regime combinations. See the report for panel b. | `summary2.json` (`PU2`, `authors_vs_median`), `ed2.pdf` and PNG |
| Extended Data Fig. 3, finite mutation | `code/fmu_sim.py`, then `code/make_ed2.py:102` from `fmu_*.json`. The exact authors'-set checks are in `finite_mu_authors.csv`, generated from `spec_*.json`. | `summary2.json` (`fmu_sims`, `finite_mu_authors`), `ed3.pdf` and PNG |
| Extended Data Fig. 4, threshold and predicate | `code/analyze2.py:262` through `code/analyze2.py:288` and `code/make_ed2.py:123`. Primary D0 level/control NPZ cooperation values and strict PU3 margins. | `threshold_sensitivity.csv`, `summary2.json` (`threshold`, `PU3_strict`), `ed4.pdf` and PNG |
| Extended Data Fig. 5, exploratory regimes | `code/analyze2.py:307` and `code/make_ed2.py:151`. D0 from `cells_primary_expl_s*.jsonl`, with primary cells as reference where plotted. | `exploratory_cells.csv`, `summary2.json` (`exploratory`), `ed5.pdf` and PNG |
| Extended Data Fig. 6, numerical diagnostics | `code/diagnostics.py` (R2), `code/analyze2.py:259` and `code/make_ed2.py:203`. GTH and historical LU solver versus the 120-digit solution, plus D2 IR audit cells. | `summary2.json` (`diagnostics`, `IR_D2`), `ed6.pdf` and PNG |

## Supplementary Information

`code/build_si.py` combines the inputs below into `supplementary.tex`. Table labels `tab:s1` through `tab:s20` identify the generated tables.

| Tables or sections | Inputs and code |
| --- | --- |
| S1–S2 | Frozen corpus and literature provenance in `code/claims.py` and `code/build_si.py`. Anchor configurations from D0 cells. |
| S3–S6 | Payoffs, IR reputation dynamics, fixation and exact finite-mutation validation from `validation2.json` (V1–V3). |
| S7–S8, Supplementary Section S5 | All primary D0 cells for S7. `dynamics_table2.csv` for S8. `summary2.json` supplies H3 and SpFI tests and the comparison with D1–D5. |
| Supplementary Section S6 | `analysis/analysis_plan_frozen.md` and `analysis/analysis_plan_amendment.md`. |
| S9–S12 | `att_table` reads Shapley and verdict-Banzhaf arrays for DR1 regimes 0 and 2, PU1 regime 0 and DR4 regime 0. |
| S13–S15 | `summary2.json` keys `threshold`, `PU3_strict` and `PU2`. |
| S16–S17 | `summary2.json` keys `variants` and `exploratory`. |
| S18–S19 | `finite_mu_authors.csv` and `summary2.json` (`fmu_sims`). |
| S20 | `summary2.json` (`diagnostics.R2_solver`) from `diagnostics.json`. |
| Supplementary Sections S12–S13 | `analysis/ssfi_minimal.py` and revision software and timing notes in `code/build_si.py`. |
