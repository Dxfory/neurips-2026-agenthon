"""Shared semantics, bounded execution and source-grounded runtime diagnostics.

Runtime task interpretation remains necessary; no task-specific answers, IDs,
templates or complete task programs are included.
"""

from .checks import Checks
from .stages import StageCache, probe
from .evidence import EvidenceLedger, availability_filter, asof_vintages
from .semantics import (SourceRef, QuantitySpec, TaskSpec, canonical_unit, convert_unit,
                        extract_task_spec, load_task_spec)
from .contracts import enforce_contract, enforce_task
from .verification import verify, verify_quantity
from .resources import ResourceBudget, measure_resources, run_batches
from .workflow import Stage, Workflow
from .documents import DocumentLimits, read_document, extract_quantity
from .artifacts import validate_artifact
from .requirements import RequirementPlan, RequirementRun, load_requirement_plan
from .normalization import (parse_panel_plan, encode_panel_label, decode_panel_label,
                            panel_artifact, load_panel_artifact, require_same_panel_axes)

__version__ = "0.2.1"
__all__ = ["Checks", "StageCache", "probe", "EvidenceLedger", "availability_filter", "asof_vintages",
           "SourceRef", "QuantitySpec", "TaskSpec", "canonical_unit", "convert_unit",
           "extract_task_spec", "load_task_spec", "enforce_contract", "enforce_task",
           "verify", "verify_quantity", "ResourceBudget", "measure_resources", "run_batches",
           "Stage", "Workflow", "DocumentLimits", "read_document", "extract_quantity",
           "validate_artifact", "RequirementPlan", "RequirementRun", "load_requirement_plan",
           "parse_panel_plan", "encode_panel_label", "decode_panel_label", "panel_artifact",
           "load_panel_artifact", "require_same_panel_axes"]
