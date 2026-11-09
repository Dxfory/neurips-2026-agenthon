# qfkit 0.6.2: explicit labelled panels

`qfkit.labelled_panel` converts a long in-memory pandas DataFrame into an explicitly ordered `(time, entity, feature)` array. All axes and missing-value policies are required arguments. The existing `numeric_time_frame`, `date_row_position` and `lagged_sample` APIs are retained unchanged.

```python
import pandas as pd
import qfkit

rows = pd.DataFrame({"period": ["later", "earlier"], "entity": ["a", "a"], "value": [4., 2.]})
panel = qfkit.labelled_panel(
    rows, time_col="period", entity_col="entity", value_cols=["value"],
    times=["earlier", "later"], entities=["a"],
    numeric_errors="raise", missing="raise",
)
assert panel["values"].shape == (2, 1, 1)
assert panel["source_positions"].tolist() == [[1], [0]]
```

The result contains float64 `values`, immutable tuples `time_labels`, `entity_labels`, `feature_labels`, integer `source_positions`, and boolean `finite_mask`. Positions refer to rows in the supplied frame, independently of its index labels. Only structurally absent rows receive `-1`; an observed or coerced NaN retains its original row position. Returned arrays do not share storage with the input.

Axes accept strings, non-bool Python/NumPy integers, finite floats up to 64 bits, and pandas Timestamp values. Integer `1`, float `1.0`, and string `"1"` remain distinct. Timestamp identity includes local timestamp and timezone name. No parsing, sorting, timezone conversion or calendar inference occurs. Durations, missing labels, booleans, compound labels and datetime values outside the supported Timestamp type raise `ValueError`. pandas may already coerce a column's scalar types during DataFrame construction; this adapter cannot reconstruct those earlier types.

Duplicate typed time/entity observations and labels outside the explicit axes raise. `numeric_errors="raise"` rejects invalid text; `"coerce"` retains it as NaN. Actual bool, complex, temporal and infinite numeric values always raise. `missing="raise"` rejects absent grid rows or numeric NaN; `"nan"` preserves the requested grid. Float64 conversion has ordinary floating-point precision limits and performs no unit conversion or rescaling.

This release makes a general input API available through package exports, the full catalog and library documentation. It does not select a sample universe, align independently chosen calendars, choose lag direction, infer units, estimate financial models, or validate task meaning. Callers must use consistent explicit axes across panels and define any shifts and estimator inputs themselves. Automatic runtime and requirements handling remain off. Engineering tests and image smoke check the API contract; model adoption, a complete input-to-result loop, and task-score improvement remain unmeasured.

The unified profile requires the 0.6.2 full-catalog identity. Historical 0.4/0.5 catalog prefixes remain byte-for-byte unchanged for legacy compatibility tests. Images and active-candidate receipts must be rebuilt and rebound by the release owner after integration; this source change does not update them.


## Fixed-count batches and mergeable statistics

`batch_slices(total, batch_size=...)` yields consecutive slices over exactly
`total` rows, including the final partial batch. It holds constant iterator
state and never draws, shuffles, omits or reseeds observations. `total=0` yields
nothing; `batch_size` must be positive. Booleans and noninteger counts raise
when the iterator is advanced.

`OnlineMoments` summarizes small vectors of reduced results without retaining
observations. `update` accepts a finite scalar-observation vector or a matrix
with **rows as observations and columns as features**. `merge` combines
nonoverlapping summaries of the same feature dimension. `result(ddof=1)` returns
`n`, `mean`, `variance`, `covariance`, and `se`; vector results always have shape
`(features,)`, including scalar observations, and covariance has shape
`(features, features)`. Standard errors are `sqrt(variance/n)` and assume
independent rows. `ddof` is required and must be smaller than `n`; no sample
variance is reported from just one observation.

```python
import numpy as np
import qfkit

rng = np.random.default_rng(17)  # one persistent generator
summary = qfkit.OnlineMoments()
for part in qfkit.batch_slices(10001, batch_size=256):
    draws = rng.standard_normal((part.stop - part.start, 6))
    left = draws.sum(axis=1)
    right = draws[:, 0] + 2 * draws[:, 1]
    # The difference uses the same row of draws before aggregation.
    summary.update(np.column_stack([left, right, left - right]))
result = summary.result(ddof=1)
assert result["n"] == 10001
paired_difference_se = result["se"][2]
```

This keeps sample count and float64 precision unchanged. The accumulator state
costs `O(features**2)`; a call to `update` uses
`O(batch_rows*features + features**2)` temporary space. It does not constrain
allocations inside a caller's simulation: reduce each bounded path block to its
small vector of required output samples, update the summary, and release the
block. Do not concatenate the blocks afterward, retain all bumped path matrices,
or interpret a time-step axis as millions of summary features.

The caller owns the exact random draw schedule. Sequential row-major draws from
one NumPy generator can be split into contiguous batches without changing their
sequence or the next RNG state. **Reordering distinct draw calls can change the
experiment**: a full array of A followed by a full array of B differs from drawing
A and B interleaved within each batch. Keep the prescribed ordering and reuse the
same batch of draws for paired scenarios. The helper never substitutes a new seed,
reduces paths or time steps, changes precision, or chooses a financial method.

For antithetic results stored as `[original half, antithetic half]`, slice by
**pair index**, align the two halves, then update with pair averages. Arbitrarily
slicing the entire 2N-vector and treating each chunk as independent changes the
variance estimate. The accumulator's count must be N independent pairs:

```python
n_pairs = len(stored_samples) // 2
if len(stored_samples) != 2 * n_pairs:
    raise ValueError("complete antithetic pairs required")
summary = qfkit.OnlineMoments()
for part in qfkit.batch_slices(n_pairs, batch_size=256):
    partner = slice(n_pairs + part.start, n_pairs + part.stop)
    summary.update((stored_samples[part] + stored_samples[partner]) / 2)
result = summary.result(ddof=1)
```

For newly generated pairs, compute both sides from the same base draw block and
form the average before `update`, avoiding storage of both full halves. For a
paired difference, update with per-row differences rather than combining the
standard errors of independent-looking summaries. Joint covariance is available
when both outputs and their dependence are needed; block standard errors must
never be averaged.

The numerical state uses float64 shifted central moments and the parallel merge
identity. Smaller floating inputs are promoted; higher-precision floating arrays,
complex/string/object/bool arrays, nonfinite data and integer values not exactly
representable in float64 are rejected. Empty updates and feature-dimension
changes raise. Arithmetic overflow or detected underflow raises without committing
a partial update; result arrays are detached from state. Floating summation order
can change final rounding bits across batch sizes or merge orders. This API checks
its computation contract, not independence, model bias or financial correctness.

These source additions do not change the declared package version. The release
owner must bind the updated full-catalog hash and rebuild the candidate before
using it; the historical catalog prefixes remain unchanged.
