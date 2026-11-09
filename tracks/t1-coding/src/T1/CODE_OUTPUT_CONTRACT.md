This complete-program response has a hard limit of 4,000 generated tokens. Produce one complete
executable program that finishes comfortably within that limit.

- Return only the code, using exactly the response format already required.
  No prose, worked derivations or restatement of the task. Omit comments and
  docstrings unless the task explicitly requires them.
- Use small top-level functions and a short main function. Factor shared operations
  instead of repeating similar loops and branches separately for each output field.
- Keep every required input read, calculation, output file and column, and finish
  with the entrypoint that runs the program. Reserve enough space for output writing
  and the entrypoint before expanding intermediate calculations.
- Use the stated input guarantees. Avoid defensive branches for conditions the
  contract already rules out. Required handling of missing or invalid observations
  remains part of the algorithm and must be implemented.
- Preserve all specified formulas, units, sample endpoints, estimator conventions,
  rounding and ordering. Shorten repetition and presentation, never the required
  mathematics or deliverables. Do not replace unfinished work with zeros, NaNs,
  placeholders, stubs, ellipses or hard-coded results.
- Before emitting the final code, check that all called functions are defined or imported and
  every required result reaches its output. Do not print this checklist or a plan.
