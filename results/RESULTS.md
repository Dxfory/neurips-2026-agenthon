# Results and evidence

## Final (sealed held-out units) — official site, read 2026-11-09
| Track | Rank | Teams | Score |
|---|---|---|---|
| T1 Coding | 48 | 61 | 0.0357 |
| T2 Forecasting | 8 | 87 (83 scored) | −0.9689 |
| T3 Simulation | 3 | 55 (54 scored) | 3,249,452.2547 |
| T4 Explainability | 26 | 63 | 0.3779 |
Evidence: https://www.agenthon.net/ (Final results tables). Prize: top-3 per track; T3 3rd place.

## T3 engineering facts (from the team's T3 poster / public repo; not organiser-certified)
- Rust rewrite (`fastsim`), 2,969 consistency scenarios, 0 row mismatches at release; 142/142
  container validations through gates g0–g3; whole image ~1.8 MB (single static binary).
- Same-machine median 880× (ABIDES simulation loop vs. fastsim whole process, 65 scenarios,
  range ~66–1,765×). This is **not** an end-to-end container speed-up or a score multiple.

## Development phase (provisional, practice feedback)
- T3: 10,000,000 (cap) under original rules; 221,088 (v18) under container-timed rules, #6/70.
- T1 #16/118, T2 #64/154, T4 #24/69 (snapshot 2026-10-15). Development scores are provisional.

Do not quote numbers that are not in this file.
