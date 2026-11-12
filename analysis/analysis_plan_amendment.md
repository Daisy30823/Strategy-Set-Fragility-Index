# Amendment log to the frozen analysis plan (revision round 1)

This log records every change made after peer review. None of the changes alters the frozen corpus, its regimes, the three pre-specified universes, the primary estimand or the three hypotheses. Each change falls into one of three classes: a correction of an implementation error, a correction of reporting, or an analysis added at revision. The added analyses are labelled as revision-stage throughout the manuscript and are not treated as confirmatory.

## A. Corrections of implementation

A1. The stationary distribution of the rare-mutation chain is now computed with the Grassmann–Taksar–Heyman (GTH) elimination algorithm, which uses no subtraction. The submitted solver, a dense LU solve of the balance equations with one equation replaced by normalisation, a least-squares fallback and clipping, returned inaccurate abundances in nearly reducible chains under strong selection (β = 10). The worst case was the complete 81-rule stern-judging chain at b/c = 3. GTH agrees with 120-digit arithmetic on these chains (Supplementary Section S11). All cells were recomputed.

A2. Random seeds are now derived from the specification of the strategy set (claim, regime, strategy-universe generator, evolutionary specification). Under the submitted scheme they were derived from the universe label. For DR3, DR4 and DR6 the authors'-encoding and intermediate universes are the same 32-strategy set, so the submitted run drew different Monte Carlo samples for identical sets. Identical sets now receive identical draws. They are reported once and enter every summary and test once.

A3. Fragility of stability claims is now computed exactly. The superset index is 1 − 2^(−q), where q is the number of omitted strategies that invade the focal resident with more than neutral fixation probability. The replacement index is 1 − C(P − q_P, k)/C(P, k). The submitted run used Monte Carlo draws for these claims.

A4. The indirect-reciprocity validation now compares the agent-based simulation with the production payoff convention. The submitted validation used a reference in which donations received included the focal individual. The agent-based runs now retain 2 × 10^6 interactions after a burn-in of 2 × 10^5; the submitted Methods described 2 × 10^6 retained interactions although only 1.8 × 10^6 were retained.

## B. Corrections of reporting

B1. Sampled SSFI values are reported with Clopper–Pearson 95% intervals. A sampled value of zero is described as "no reversal in n sampled supersets", not as a proof of robustness. Under the pre-specified estimand the empty extension is included, so SSFI ≤ 1 − 2^(−m) < 1, and values that would round to 1 are shown as > 0.999.

B2. D1 to D5 are described as alternative evolutionary specifications (two selection strengths, one alternative update rule and two population sizes), not as alternative update rules.

B3. The pre-specified Wilcoxon test for H3 is still reported, but it treats parameter regimes of the same claim as independent. Claim-level inference (an exact sign-flip test on claim means and a claim-clustered bootstrap interval) is reported alongside it. The same claim-level treatment is applied to the exploratory margin association.

B4. "No difference" for H2 is replaced by "not supported; the interval is wide".

## C. Analyses added at revision (not pre-specified)

C1. SML-equivalence (clone) quotient of every universe. Two strategies are merged when they are payoff-neutral with each other in every two-strategy composition, earn identical payoffs against every other member of the universe, receive identical payoffs from every other member, and have the same homogeneous cooperation rate. In the rare-mutation limit such strategies are interchangeable, and a uniform mutation kernel over labels gives them extra weight.

C2. Inclusion-probability sensitivity: SSFI for q ∈ {0.05, 0.1, 0.25, 0.5, 0.75}. Size-stratified SSFI(k) for extensions of exactly k strategies, with k ∈ {1, 2, 4, …, 128, m}. A list of the omitted strategies whose single addition reverses the verdict.

C3. Attribution. Margin Shapley values are now estimated from 1,000 permutations, with Monte Carlo standard errors and split-half rank stability. Omitted strategies are placed in three categories (neutral-entry, neutral but not entry, non-neutral), so that the category sums add up to the total change in margin. A verdict-reversal Banzhaf index is added: the probability of reversal when a strategy is included minus the probability when it is excluded.

C4. Two additional evolutionary specifications. D6 is a local mutation kernel with weights exp(−λd), where d is the distance between strategy encodings. D7 sets a common population size N = 50 in all families.

C5. A specification multiverse for the authors' sets: 300 random specifications per claim–regime, drawn over update rule, selection strength and population size. The authors'-set verdict is also evaluated under finite mutation (μ = 10^−3, 10^−2) by solving the full stationary distribution exactly, for the direct- and indirect-reciprocity claims.

C6. Model-convention variants:
- smallest instead of largest root of the reputation map;
- reputation dynamics with the recipient drawn from the other N − 1 individuals;
- punishment only in groups where the public-goods game takes place.

C7. Exploratory claim variants, reported separately and excluded from every corpus summary:
- stern judging at b/c ∈ {2.2, 2.4, 2.6, 2.8}, and at b/c = 2, which was screened out before freezing;
- simple standing, which was also screened out before freezing;
- the WSLS claims at δ = 0.99.

C8. Finite-mutation individual-based simulations in the 32- and 243-strategy direct-reciprocity universes. Multiplicity diagnostics for the roots of the reputation map.

C9. Threshold and predicate sensitivity: the cooperation threshold of level claims varied from 0.4 to 0.6; PU3 evaluated under the strict predicate that punishers are more abundant than every other strategy in the set; PU2 replacement and superset indices reported side by side.

C10. Exploratory comparisons of the authors' sets with the median of the superset multiverse (margins and cooperation), at the level of distinct model-regime combinations.

## D. Numbers that changed relative to the first version

D1. Mean superset SSFI in the encoding universe: 0.106 in the first version, 0.107 now (duplicated universes of DR3, DR4 and DR6 now share draws). The intermediate (0.141) and largest-universe (0.180) means and the counts of reversed claim-regimes (10 of 47; claims DR1, DR3, DR4, DR6, PU1) are unchanged.

D2. DR1 at δ = 0.8, largest universe: SSFI 0.542 in the first version, 0.543 now. DR3 at δ = 0.9, 32-strategy universe: 0.979 then, 0.976 now. DR6: 0.87 to 1.00 then, 0.875 to 1 − 2^(−q) now (exact). PU1 in the largest universe: 0.48 to 0.62 then, 0.48 to 0.63 now. Largest SSFI of the image-scoring control IR2: 0.043 then, 0.042 now.

D3. Indirect reciprocity under strong selection (D2): the first version reported verdicts computed with the inaccurate LU solver; with GTH, IR1 and IR3 at b/c = 3 reverse in the largest universe under D2.

D4. The Shapley attribution of PU1 now identifies DpL and LpC, not CpL, as the main drivers; the first version described the neutral-entry strategy CpL as the source of the reversal.

D5. The pre-specified row-level Wilcoxon test for H3 is now reported only for continuity; the claim-level sign-flip test (P = 0.016) is primary.
