# Simulation protocol

## Configurations and initial conditions

`code/claims.py` defines each authors' set, core strategy list, verdict, universe generator and parameter regime. `code/engine.py` defines the payoff and fixation functions. Table S1 in the generated `supplementary.tex` enumerates the corpus and literature sources. Table S2 reports the three anchor configurations.

| Family | D0 inputs | Strategy universes |
| --- | --- | --- |
| Direct reciprocity (DR) | Population 20, repeated prisoner's dilemma payoffs `(R,S,T,P)=(3,0,4,1)`, continuation probability and execution error by claim. Encodings include the first-round action probability. | 8 reactive, 32 deterministic memory-one or 243 memory-one grid strategies. DR3, DR4 and DR6 use the same 32 strategies for encoding and intermediate. |
| Optional public goods with punishment (PU) | Population 100, group size 5, contribution cost and loner payoff both 1. The four regimes vary the multiplier and punishment cost, with fine 1. Groups are drawn without replacement. | 9, 12 and 24 strategies, in order of increasing participation and punishment options. |
| Indirect reciprocity (IR) | Population 50, donation cost 1, execution and assessment error probabilities both 0.02. The norm and donation benefit are specified by regime. D0 uses the largest reputation root and excludes self-interaction from donations received. | 4, 16 and 81 rules. |

The primary small-mutation calculation solves for the stationary distribution over monomorphic states. Fixation starts with one mutant in a resident population. The stationary calculations solve for abundances directly. The evolution specifications are defined in `code/claims.py`.

| ID | Update, selection and population |
| --- | --- |
| D0 | Fermi pairwise comparison, beta 1, baseline population sizes and uniform mutation |
| D1, D2 | Fermi with beta 0.1 and 10, respectively |
| D3 | Moran updating with selection strength 1 |
| D4, D5 | Fermi at half and twice the baseline population size, respectively |
| D6 | Fermi with local mutation weights `exp(-lambda * d)`. DR and IR use L1 encoding distance and lambda 3. PU uses Hamming distance and lambda 1.5. See `code/audit2.py:92`. |
| D7 | Fermi with population size 50 in every family |

Parameters in D1–D7 follow D0 for settings omitted from the table. The main level threshold is 0.5, with verdict margin tolerance `1e-9`. The stability predicate admits fixation up to the neutral rate plus `1e-12`.

## Enumeration, sampling and seeds

- **SSFI.** For non-stability claims, enumerate all supersets with at most 16 omitted strategies. For larger sets draw 20,000 independent masks with inclusion probability 0.5. For stability claims use the exact invasion count and the closed-form indices. Replacement enumerates up to 20,000 admissible sets or samples 20,000 when the count is larger, with a separate closed form for stability. See `code/audit2.py:302` through `code/audit2.py:351`.
- **Revision sensitivities and attribution.** D0 estimates SSFI at inclusion probabilities 0.05, 0.10, 0.25 and 0.75 using 4,000 masks when sampled. Fixed-size strata enumerate at most 2,000 combinations or sample 2,000 for each applicable size from 1, 2, 4, …, 128 and the full omitted set. The clone-quotient calculation enumerates through 16 omitted classes and samples 5,000 masks above that size. Attribution uses 1,000 random permutations for margin Shapley values. See `code/audit2.py:353` through `code/audit2.py:462`.
- **Audit seed.** `seed_of` applies CRC32 to UTF-8 pipe-joined claim ID, zero-based regime index, universe generator, dynamics ID and variant. The quotient appends `q`, inclusion sensitivity appends `qs|<q>`, and size sensitivity appends `k|<k>`. Identical universe labels share the computation through `dup_of`. The primary random stream also supplies replacement draws when needed and attribution orderings.
- **Evolutionary specification multiverse.** For each claim and regime, draw 300 combinations. Select Fermi or Moran with equal probability, draw strength log-uniformly on [0.1, 10], and draw integer population size uniformly from half to twice the family baseline, inclusive. The seed is CRC32 of `spec|<claim>|<regime>`. For eligible DR and IR level, control and winner claims, solve the exact finite-mutation stationary system on the authors' set at mutation probabilities 0.001 and 0.01. See `code/spec_multiverse.py`, `finite_mu_verdict` and `run`.
- **Individual-based finite mutation.** DR1 at continuation probabilities 0.8 and 0.9, and DR4 at 0.8 with execution error 0.01, each use the 32- and 243-strategy universes and mutation probabilities 0.01, 0.001 and 0.0001. Each replicate starts monomorphic at a uniformly selected strategy, with population 20 and Fermi beta 1. Seeds are 1000–1003 per case. Each replicate runs 80 million steps at 0.01 and 800 million at the other two rates, with a 10% burn-in and cooperation sampled every 97 steps. The cap in `code/fmu_sim.py` determines the implemented step counts.
- **Validation and inference.** The validation RNG starts at 20261105. The reputation validation starts with all reputations good and uses the pairwise compositions in `code/validate2.py:136`, with 200,000 burn-in interactions followed by 2 million retained interactions. Simulated fixation starts with one mutant. The inference RNG starts at 20261105 and uses 4,000 claim-cluster bootstrap resamples per call. The manuscript reports 2,000. The plotted H1 jitter has seed 1. See the [packaging report](PACKAGING_REPORT.md) for reporting differences.

## Full execution order

`reproduce.py full` runs the following commands in a fresh results directory, with the paths below shown from the package root. Substitute an existing results directory for `OUT` and each shard index for `i`. Shards write disjoint files. For manual execution, set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` to 1 per process. Use `MPLBACKEND=Agg` on a headless machine.

| Step | Command | Shard indices |
| --- | --- | --- |
| Primary audit | `python code/audit2.py OUT 1 --variant primary --shard i --nshards 16` | 0–15 |
| Reputation equilibrium | `python code/audit2.py OUT 1 --variant ir_smallest_root --light --shard i --nshards 3` | 0–2 |
| Reputation self-exclusion | `python code/audit2.py OUT 1 --variant ir_full_selfexcl --light --shard i --nshards 3` | 0–2 |
| Punishment convention | `python code/audit2.py OUT 1 --variant pu_punish_if_game --light --shard i --nshards 3` | 0–2 |
| Exploratory regimes | `python code/audit2.py OUT 1 --exploratory --light --shard i --nshards 4` | 0–3 |
| Specification multiverse and exact finite mutation | `python code/spec_multiverse.py OUT/spec_i.json i 4` | 0–3 |
| Individual-based finite mutation | `python code/fmu_sim.py OUT/fmu_i.json i 3` | 0–2 |
| Validation and diagnostics | `python code/validate2.py OUT/validation2.json` and `python code/diagnostics.py OUT/diagnostics.json` | One invocation each |
| Analysis | `python code/analyze2.py OUT` followed by `python code/table1.py OUT` | After all simulations finish |
| Figures and supplement | `python code/run_figures.py OUT` and `python code/build_si.py OUT` | After analysis |

The three convention variants evaluate D0. The exploratory invocation retains D0–D7 with the light settings and feeds only its D0 cells into the reported exploratory summaries. Preserve the original shard counts for bootstrap input ordering. `reproduce.py stored` supplies the raw inputs from the archive and then performs the last two rows of the workflow.

The authors report about 30 minutes for the primary audit with 16 workers, 25 minutes for the specification multiverse with four workers, and 25 minutes for the individual-based simulations with three workers on a 20-core Windows workstation. These are timings for separate stages. The full command launches 38 processes in total with at most `--jobs` running at a time. Set the worker count for the available memory and physical cores. Allow several minutes even for the stored workflow, which recomputes the clustered bootstrap.
