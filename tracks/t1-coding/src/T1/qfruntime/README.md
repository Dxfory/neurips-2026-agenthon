# qfruntime 0.2.1

Reusable runtime checks and provenance tools for the current task's data. This
package contains no task IDs, task routing, reference answers or complete task
solutions. It is separate from qfkit's numerical function catalog.

## Shared task meanings

`QuantitySpec` records a quantity's name, unit, shape, axes, method, time basis,
timezone, tolerances and reporting period. `SourceRef` attaches an exact Unicode
character span and quote to the definition. `TaskSpec` groups these definitions,
unresolved requirements and source hashes; use the same definitions in parsing,
contracts, dependency stages and numerical verification. Their fingerprints bind
the declared meanings, rather than just a quantity's name.

`extract_task_spec(text)` recognises explicit declarations such as
`quantity rate: unit=percent; shape=[]; axes=[]; method=literal; time_basis=static`
and simple statements such as `rate is measured in percent.` Unsupported prose
remains in `TaskSpec.unresolved`. Missing fields remain unknown. This is not an
arbitrary natural-language interpreter. `TaskSpec.from_dict(payload,
sources={source_id: original_text})` verifies source spans and content hashes;
that verifies attachment, not whether the interpretation follows from the text.

`canonical_unit` normalises spelling aliases. `convert_unit(values, from_unit,
to_unit)` performs explicit fraction/percent/basis-point conversions. Currency,
percentage-point differences, methods and time conventions are not inferred.
Time bases are `static`, `event`, `available`, `revision`, `decision` and
`period_end`; timezone and reporting period are separate fields.

## Six capabilities

| Capability and public API | Implemented behavior | Boundary |
|---|---|---|
| 1. Calling contracts: `enforce_contract`, `enforce_task`, `Checks` | Checks declared units, rectangular shapes, axes, methods and time meanings against actual values; preserves unknown and conflicting definitions separately | Complete contracts require explicit definitions and source binding. No implicit conversion or broadcasting. A source quote does not prove semantic interpretation. |
| 2. Numerical verification: `verify`, `verify_quantity` | Routes supported integrals, derivatives, linear systems, sums, cashflows and probabilities to bounded reference or consistency checks | Cross-method references use error estimates, not rigorous universal error bounds. Unknown methods, exhausted reference budgets and insufficient separation from reference error stay unverified. Residuals and sums are consistency checks, not independent solutions. |
| 3. Small trials and resources: `probe`, `measure_resources`, `run_batches`, `ResourceBudget` | Measures a small trial, wall/CPU time and RSS; executes bounded batches with a local RNG and preserves declared sample and error targets | Timing extrapolation is an estimate. RSS is the process-lifetime high-water mark, not incremental callback memory. Batch budgets are cooperative, checked between batches; they cannot interrupt a callback. |
| 4. Dependency stages: `Stage`, `Workflow`, `StageCache` | Validates a DAG, orders ancestors, blocks descendants of failures and reuses validated JSON intermediates; binds input content, parameters, versions, semantics and compute/validator fingerprints | Caller must declare every input and include file content or its hash. Opaque callable dependencies disable reuse. Serial execution only; no cross-task answer cache. |
| 5. Document and time evidence: `read_document`, `extract_quantity`, `EvidenceLedger`, `availability_filter`, `asof_vintages` | Extracts explicit text/HTML/CSV/JSON fields with exact source spans, locators and hashes; retains duplicates and conflicts; checks explicit availability and revision cutoffs | No PDF/OCR or arbitrary prose extraction. No unit/date guessing. A located value is not an independently verified fact. Event time is not publication time. |
| 6. Transfer validation: synthetic component tests and frozen composition cases | Exercises changed fields, scales, dates, file contents, dependencies, semantic definitions and injected numerical errors across the public APIs | Local engineering evidence does not establish real-task adoption, House acceptance or final-evaluation score gains. See the validation section below. |

The checks and evidence tools use Python's standard library. The optional panel
packet tools below require the installed qfkit 0.6.2, NumPy and pandas. This is an
added package; the controller does not offer it to tasks explicitly restricted
to the standard library.

## Explicit panel packets

`parse_panel_plan` binds a closed `panel-input-plan-v1` proposal to one CSV's
declared column order, exact current TASK spans and caller-supplied source and
semantic hashes. Time and entity axes are ordered typed literals. String,
integer, float and roundtrippable Timestamp identities stay distinct. The
proposal is limited to 8 KiB of UTF-8; it never sorts, infers axes or runs code.

`panel_artifact(frame, binding)` applies qfkit's `labelled_panel` to that frame and
returns a finite JSON manifest plus fixed numeric bytes. Values, typed labels,
source row positions and finite masks survive loading; an observed NaN retains
its row position, while a structurally absent row has position -1. The manifest
limit is 32 KiB, checked before return. The payload limit is 8 MiB, checked
before panel allocation.
The numeric layout has no object arrays, pickle or executable headers.

`load_panel_artifact(..., expected_binding=..., expected_manifest_sha256=...)`
requires binding and digest held independently by the caller. It checks packet
identity, axes, layout, values, masks and row coverage. `require_same_panel_axes`
compares actual typed time/entity members and order across panels; equal shape
alone is insufficient. Feature counts and each file's row positions may differ.
`encode_panel_label` and `decode_panel_label` expose the same bounded label codec.

All panel entry points reject installed qfkit version drift. A version is a
compatibility gate, not an attestation of algorithm bytes or numeric correctness.
The receipt deliberately says `source_dataframe_link=caller_asserted` and
`semantic_verification=unverified`: this API does not read CSV files or establish
that a supplied frame came from the declared file. It does not prove downstream
adoption. Large axes can exceed the literal proposal limit; a compact selector
and trusted CSV node remain separate work. Model proposals must also fit the
existing model reply budget; 8 KiB is not an extra allowance.

These APIs are installed but the controller does not execute panel plans yet.
No automatic runtime/requirements mode or additional model request is enabled.

## Controller integration

### Requirements and output structure (experimental)

`RequirementPlan` accounts for the complete current instruction, including text
the narrow `extract_task_spec` grammar cannot interpret. It divides paragraphs
into exact source spans of at most 500 characters; all blocks begin unassigned.
`load_requirement_plan()` reads a bounded, hash-checked, read-only source context
from `T1_REQUIREMENT_SOURCE`, independently of semantic extraction success.
It does not infer requirements or replace natural-language interpretation.

`plan.add(id, source_ids=[...], inputs=[...], method=..., stages=[...],
checks=[...], artifacts=[...], quantities=())` registers the current model's
interpretation. `plan.context(ids, reason=...)` explicitly labels contextual
material. Inputs are provenance labels, not proof that input data were validated.
Optional `RequirementPlan(text, semantics=task_spec)` binds source-checked quantity
definitions; referenced quantities also need complete definitions and passing
semantic contract checks. Freeze the plan before constructing `RequirementRun`.

`run.stage(name, callback)` records whether a required computation callback ran
and completed in this attempt. `run.finish(checks, output_root)` inspects actual
artifacts and connects them with those receipts and actual Checks records.
Missing stages/checks/files and existing failed diagnostics remain incomplete;
unassigned text, incomplete definitions and skipped checks remain unverified.
It emits bounded, hashed coverage diagnostics through Checks and writes a final
completion record only after every declared requirement has been visited. The
controller checks its counts and plan identity; an interrupted report cannot
claim complete coverage. Repair cannot silently remove requirements or change
their specification. Legitimate definition revisions still need a future
explicit revision/revalidation protocol.

`covered` describes the declared steps/checks/schema; it does not establish that
all requirements were interpreted correctly, that callbacks implement the chosen
method, or that numerical checks are bound to the emitted file values. It is not
an official score. Whole-task adoption and pass@1 remain separate experiments.

`validate_artifact(root, contract)` independently checks a relative regular file
without modifying it. JSON supports object properties/required keys, array items
(every element), strict scalar types and explicit nullable. CSV supports an exact
column set, optional order, per-column scalar types and row count; it currently
uses comma-separated UTF-8 with a header. Unsupported conventions are unverified,
not guessed. File hashes cover the complete inspected bytes. Reads default to
4 MiB, with an explicit limit up to 64 MiB; schema/content depth, field sizes and
diagnostic counts are bounded, and symlinks/path traversal are rejected. Format
checks do not prove numerical correctness, sorting or cross-file relationships.

Add `T1_REQUIREMENTS=1` alongside `T1_RUNTIME=1` to offer this protocol to the
model. Both are off by default. The protocol adds guidance to the existing code
generation request, not a mandatory extra House call. Research matrices and
public checker mappings are never loaded by the runtime. Source inspection and
schema checks use local compute; model interpretation still consumes tokens.

Set `T1_RUNTIME=1` inside the solver container (an explicit Docker
`--env T1_RUNTIME=1`, or an experimental image configuration) to offer these tools
and consume their diagnostics. Merely setting it in the host shell does not make
the local Docker runner forward it. It is experimental and off by default. Historical
request prompts remain unchanged when the flag is off. Both Docker builds install
the package; the source-only build excludes this README, tests and task data.

The controller creates one temporary cache per solve and a new report path per
program run. `Checks()` reads `T1_RUNTIME_REPORT` automatically. The cache root,
namespace and remaining per-run time are supplied via `T1_RUNTIME_CACHE`,
`T1_RUNTIME_NAMESPACE` and `T1_RUNTIME_BUDGET_SEC`. When extraction succeeds,
`T1_TASK_SPEC` points to a controller-private, read-only context containing the
original instruction and its source-bound `TaskSpec`. `load_task_spec()` verifies
that context; its unresolved requirements still need interpretation. Controller
status records only the semantic state, quantity/unresolved counts and definition
fingerprint, not the instruction text. These scratch files are removed at solve
end and are not deliverables. Real generated code must run in the dedicated
container; enabling helpers does not relax the worker's execution isolation.

The worker launcher also writes `T1_RUNTIME_USAGE`. This bounded private diagnostic
contains only fixed capability call counters: checks, semantics, contracts,
verification, resources, workflow, documents, evidence and cache. It contains no
arguments, source contents, generated code or numerical outputs. Importing a
helper is not a call; one operation can increment several counters through nested
helpers. Abrupt termination can leave lower-bound counts. A call count does not
establish that a check passed, a result was adopted or the task was correct. Usage
states `valid`, `missing` and `invalid` remain separate from check results.

A report can contain at most 128 records and 256 KiB. Writes are atomic and reject
symlinks. The controller independently reads a bounded regular file, rejects
duplicate keys, nonfinite numbers and contradictory summaries, and exports only
fixed categories, numeric summaries and hashes. `missing`, `invalid`, `empty`,
`failed`, `skipped` and `unverified` remain distinct from passing evidence.

The existing guarded repair flow handles numerical failures, including Stack12
when its older self-check path is disabled. A repair must retain every previous
check's ID, kind, parameters and evidence declaration. Previously passing checks
cannot regress; failed checks cannot become skipped or acquire a larger effective
tolerance. Dropping the report or adding new failures rejects the repair. Output
files and diagnostic state are restored together. This also applies to ordinary
crash and timeout repair, not only post-run review.

This is deliberately conservative: changing a contract's declared unit/method,
or increasing a Monte Carlo standard-error allowance, can reject an otherwise
legitimate correction. Correct conversions should occur before checks expressed
in stable target units. These are self-reported diagnostics; the controller cannot
prove the checks cover all outputs, establish method independence, or prevent a
program from fabricating internally consistent values. It never marks an official
score or official acceptance from these reports.

`enforce_task` derives check IDs from the complete task fingerprint and quantity
name, so different definitions can coexist in one `Checks` session. Repeating the
same task definition in that session still triggers duplicate-ID protection.
Earlier failed records remain present. Changing a task definition during repair
does not evade the controller guard: it changes IDs, so replacing an old check is
still treated as removing the original check.

## Minimal runnable example

Install the package, or run these examples from `T1` with `PYTHONPATH=qfruntime`.
This example supplies all values and uses the same quantity definition for its
contract and numerical reference. Outside the controller, `Checks()` returns an
in-memory report unless an explicit report path or `T1_RUNTIME_REPORT` is supplied.

```python
from dataclasses import replace
from qfruntime import Checks, enforce_task, extract_task_spec, verify_quantity

instruction = (
    "quantity area: unit=dimensionless; shape=[]; axes=[]; method=analytic; "
    "time_basis=static; atol=1e-8; rtol=1e-8"
)
task = extract_task_spec(instruction)
quantity = task.get("area")
value = 8 / 3  # Analytic integral of x**2 from 0 to 2.
checks = Checks()
try:
    enforce_task(checks, task, {"area": value},
                 {"area": replace(quantity, source=None)})
    result = verify_quantity(
        checks, "area_reference", quantity=quantity, actual=value,
        operation="integral",
        inputs={"function": lambda x: x**2, "lower": 0, "upper": 2},
    )
    assert result["status"] == "passed"
finally:
    report = checks.finish()
```

Known contract failures record a failed check and raise `ValueError`; incomplete
contracts return `unverified`. Basic numerical `Checks` methods record failure and
return `False`, preserving the opportunity to write normal deliverables. The
`verify` adapters return structured `passed`, `failed` or `unverified` results;
unsupported or uncertain references produce skipped diagnostics. `verify_quantity`
takes method and tolerances only from the supplied quantity, without independent
override arguments. Different method labels alone do not prove independence.
Numeric inputs that cannot be represented safely are rejected rather than silently
rounded into a passing comparison.

## Documents and intermediate stages

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from qfruntime import (
    QuantitySpec, Stage, StageCache, Workflow, extract_quantity, read_document,
)

def source(inputs, parameters, dependencies):
    return inputs["value"]

def convert(inputs, parameters, dependencies):
    from qfruntime import convert_unit
    return convert_unit(dependencies["source"], "basis_point", "fraction")

def valid(value):
    return type(value) in (int, float)

with TemporaryDirectory() as directory:
    root = Path(directory).resolve()
    (root / "input.csv").write_text("amount,unit\n25,bps\n", encoding="utf-8")
    quantity = QuantitySpec("rate", "basis_point", shape=(), axes=(),
                            method="literal", time_basis="static")
    document = read_document(root, "input.csv")
    extracted = extract_quantity(document, quantity,
                                 selectors={"value": "amount", "unit": "unit"})
    assert extracted["status"] == "extracted"
    stages = [
        Stage("source", source, valid, version="1",
              inputs={"value": extracted["value"], "sha256": document.sha256},
              semantic_spec=quantity),
        Stage("fraction", convert, valid, version="1", dependencies=("source",)),
    ]
    workflow = Workflow(stages, cache=StageCache(root / "cache", namespace="example"))
    result = workflow.run(targets=["fraction"])
    assert result["status"] == "completed"
    assert result["results"]["fraction"] == 0.0025
```

`read_document` accepts only bounded local regular files beneath the supplied root,
rejects symlinks/path traversal and does not decompress archives. `DocumentLimits`
bounds bytes, records, fields, nesting, candidates and emitted provenance; limits
fail explicitly rather than truncating. Text requires labelled fields, HTML uses
explicit headers or labelled text, CSV uses a comma/quote dialect and JSON retains
duplicate fields. HTML is not a CSS renderer; explicit hidden content and merged
cells need preprocessing. PDF and OCR are unsupported.

`extract_quantity` returns every candidate and its exact source spans, locators and
hash. Only a unique compatible scalar becomes `extracted`; duplicate or uncertain
facts remain `ambiguous` or `unverified`, and future vintages are `excluded`.
`method="literal"` means reading a stated number; other method interpretations and
array assembly require separate work. Explicit aware timestamps are normalised to
UTC with their original text and offset retained. A named source timezone is not
inferred from an offset; declared zones other than `UTC`, or time metadata without
an explicit time basis, remain unverified here. Use `as_of` with declared availability/revision selectors;
set `revision_at=None` explicitly only for an unrevised stream. The record-oriented
`availability_filter` and `asof_vintages` instead name that argument
`revision_field=None`. `EvidenceLedger` retains conflicting facts and exact quotes
of at most 500 Unicode characters; attachment is not factual verification.

A `Stage` callback receives `(inputs, parameters, dependencies)` and returns finite
JSON. Its validator must return literal `True` without mutation, including on cache
hits. Input, parameter, semantic or upstream signature changes invalidate affected
descendants while independent branches can reuse. Failed stages block descendants
even when old successful intermediates exist. `Workflow` reports `completed` only
for validated requested computations; that status is not task acceptance. Cache
files use bounded JSON, atomic writes and no pickle/eval. Direct `StageCache.run`
remains available when the caller manages dependency order.

## Resource measurements and batching

```python
from qfruntime import ResourceBudget, measure_resources, probe, run_batches

trial = probe(lambda n: sum(range(n)), sample_size=20, target_size=200,
              time_budget=1, growth_power=1)
measured = measure_resources(lambda: sum(range(200)))
batch_result = run_batches(
    lambda count, rng: [rng.random() for _ in range(count)],
    max_samples=40, batch_size=10, seed=17,
    budget=ResourceBudget(wall_seconds=1),
)
assert measured["result"] == 19900
assert batch_result["max_samples"] == 40
# Inspect batch_result["status"] and ["reason"] before using its estimate.
```

`probe` executes `compute(sample_size)` exactly once and never executes the full
target. It extrapolates time using the declared growth power and fixed cost;
requested memory remains unverified. `measure_resources` records wall time,
process CPU time and process-lifetime peak RSS when available. An RSS reading is
not the current footprint or the callback's incremental memory.

`run_batches` passes a local `random.Random` to `draw(count, rng)` without changing
global RNG state. Invalid batches commit no samples. Fixed-count mode requires all
`max_samples`; with `absolute_error`, stopping also requires
`required_min_samples` and the original estimated error target (the default minimum
is `max_samples`). The normal-SE estimate assumes IID finite-variance samples;
optional stopping, dependence, heavy tails and bias prevent a general coverage
guarantee. Budgets include callback and aggregation time but cannot preempt either.
Zero budgets stop before drawing; exceeding a budget returns `incomplete`, even
if a final batch produced enough samples. Unavailable RSS cannot satisfy a memory
budget. Precision and sample requirements are never silently relaxed.

## Validation and remaining work

`runtime_tests` contains synthetic component and transfer tests, including scale,
schema, date, timezone, file-content, dependency and injected numerical errors.
Controller tests exercise actual handwritten subprocesses and scripted repair
responses. Offline container smoke checks package and worker availability.

Six newly frozen composition cases first passed 5/6. The remaining case exposed
check-ID collisions between different task definitions; after repair, the exact
same cases passed 6/6 as a regression. The first result and frozen cases were
retained. This is not a new blind evaluation: the same implementation pass
wrote the composition cases and the document extractor, and had inspected the
shared semantic module. Other new modules were exercised from their public
API contracts. The retained local audit
is excluded from the runtime package; see the
[six-capability validation account](../verification/six-capabilities-results.md)
for its evidence locations and qualification boundaries.

These checks establish engineering behavior, not task accuracy or score gain.
Arbitrary natural-language interpretation, unrestricted raw-document parsing,
PDF/OCR, full-workload resource guarantees and universal independent-reference
selection remain outside the implemented scope. Real House adoption and paired
performance still need separate evidence before enabling the runtime by default.
Earlier receipts, including `../verification/runtime-validation.json`, describe
their frozen versions and should not be treated as validation of every 0.2 change.
