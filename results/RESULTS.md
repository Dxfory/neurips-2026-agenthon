# Results and evidence

## Final, sealed held-out units, read 2026-11-09

**Final, sealed held-out units, read 2026-11-09: T3 3/55, T2 8/87, T4 26/63, T1 48/61.**

| Track | Rank | Teams | Score |
|---|---|---|---|
| T3 Simulation | 3 | 55 (54 scored) | 3,249,452.2547 |
| T2 Forecasting | 8 | 87 (83 scored) | −0.9689 |
| T4 Explainability | 26 | 63 | 0.3779 |
| T1 Coding | 48 | 61 | 0.0357 |

Evidence: https://www.agenthon.net/ (Final results tables). Prize: top-3 per track; T3 3rd place is the T3 Prize.

## T3 engineering facts (team poster and the public fastsim repository)

- Rust rewrite (`fastsim`). **2,969** consistency scenarios, **0** row mismatches at release. **142/142** container validations through gates g0–g3. Whole image **1.8 MB** (single static binary).
- **880×** is the same-machine median of the ABIDES simulation loop against the fastsim whole process, 65 scenarios, range about 66–1,765×. The official Final score is **3,249,452.2547**.
- **45** Rust tests recorded for fastsim on the team portfolio page, alongside the 2,969 scenarios with 0 row mismatches, the 142/142 container validations through gates g0–g3, and the 1.8 MB image. Portfolio page: https://github.com/Dxfory/Dxfory/blob/main/portfolio/agenthon.md

## Development snapshot (practice feedback, 2026-10-15)

The Final table above is the public lead. The lines below are the development snapshot from 2026-10-15.

- T3: 10,000,000 (cap) under original rules; 221,088 (v18) under container-timed rules, #6/70.
- T1 #16/118, T2 #64/154, T4 #24/69 (snapshot 2026-10-15).

The figures to quote from this file are the Final table, the T3 engineering facts, and this labeled development snapshot.
