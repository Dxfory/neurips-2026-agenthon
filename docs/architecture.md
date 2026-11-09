# Architecture

```mermaid
flowchart LR
  subgraph Sandbox["Official sandbox (no internet)"]
    IN[/task dir/] --> AG[Track agent]
    AG -->|restricted| HOUSE[(House model endpoint)]
    AG --> OUT[/output/]
  end
  OUT --> G0[g0 integrity] --> G1[g1 schema] --> G2[g2 cutoff/resource] --> G3[g3 domain semantics] --> SC[Track scorer]
```

**T1 thin loop:** contract request → code request → isolated execution → feedback → repair,
with a hard request cap (≤25), atomic acceptance of model code, best-attempt retention.

**T2:** numeric forecaster; each change is a pre-registered experiment with a held-out test window.

**T3:** Rust `fastsim`; differential check against the Python reference on every scenario.

**T4:** code computes facts first → House model judges → cited rationale; per-task cutoff policy
(`artifact_policy.py`) decides which fitted artefacts may be used.
