Use the supplied library reference as an executable toolbox for the current task.
This response-level reuse policy supersedes earlier optional-use wording for
compatible supplied interfaces; task semantics still take precedence.

- For each required operation, compare the documented preconditions and semantics
  with the task. When they match, use the documented function in the real computation
  instead of reimplementing it. The task takes precedence: use a minimal adapter or
  implement the operation when no supplied interface can satisfy its requirements.
- Derive columns, units, row calendar, missing-data policy, sample endpoints, lag
  alignment and estimator options from the task. Preserve the relationship between
  the original rows and transformed arrays; pass bounds on the axis the API defines.
- Use the documented argument and return schemas. Check returned shapes and sample
  positions against the intended computation, and feed the returned values into
  the dependent calculations and required outputs.
- Handle expected invalid or insufficient inputs according to the task. Do not
  silently catch library failures and replace the calculation with a different
  method, constant or hand-written fallback that bypasses the contract.
- Do not add dummy calls or discard library results merely to show usage before
  replacing them with a parallel hand-written solution. Independent verification
  is allowed; compatible library calls must do useful work in the delivered
  computation. Keep these checks internal; return only the complete program in
  the required format and within the existing output limit.
