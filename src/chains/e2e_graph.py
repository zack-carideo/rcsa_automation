"""End-to-end RCSA Control Description QC pipeline using LangGraph."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import sys
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from prompts.registry import PromptRegistry
    from configs.config import RunConfig
    from chains.pptx_builder import build_qc_pptx
else:
    from ..prompts.registry import PromptRegistry
    from ..configs.config import RunConfig
    from .pptx_builder import build_qc_pptx

load_dotenv()

# ── Tracing (active when LANGCHAIN_API_KEY is set) ──────────────────────
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "RCSA")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


# ── Pydantic models ─────────────────────────────────────────────────────
class ControlRecord(BaseModel):
    risk_description: str = Field(min_length=1)
    control_description: str = Field(min_length=1)


class QCResult(BaseModel):
    record: ControlRecord
    qc_output: str
    overall_assessment: str  # MEETS | PARTIALLY MEETS | DOES NOT MEET
    pptx_path: str = ""


# ── Graph state ──────────────────────────────────────────────────────────
class GraphState(TypedDict, total=False):
    config: RunConfig
    records: list[ControlRecord]
    qc_results: list[QCResult]
    summary: dict


# ── Helpers ──────────────────────────────────────────────────────────────
def _parse_overall(text: str) -> str:
    """Extract Overall Assessment rating from QC markdown output."""
    m = re.search(
        r"Overall\s+Assessment\s*\|\s*(MEETS|PARTIALLY MEETS|DOES NOT MEET)",
        text, re.IGNORECASE,
    )
    return m.group(1).upper() if m else "DOES NOT MEET"


def _get_config(state: GraphState) -> RunConfig:
    return state.get("config") or RunConfig()


# ── Node functions ───────────────────────────────────────────────────────
def validate_input(state: GraphState) -> dict:
    """Validate all input records via pydantic."""
    return {"records": [ControlRecord.model_validate(r) for r in state["records"]]}


def evaluate_controls(state: GraphState) -> dict:
    """Run QC evaluation on each control record."""
    cfg = _get_config(state)
    registry = PromptRegistry(_PROMPTS_DIR / "registry.yaml")
    prompt, meta = registry.get("e2e")

    model = cfg.model or meta["model"]
    temperature = cfg.temperature if cfg.temperature is not None else meta["temperature"]
    llm = ChatAnthropic(model=model, temperature=temperature)
    chain = prompt | llm

    results = []
    for rec in state["records"]:
        resp = chain.invoke({
            "risk_description": rec.risk_description,
            "control_description": rec.control_description,
            "qc_standards": meta["qc_standards"],
            "good_control_example": meta["good_control_example"],
            "bad_control_example": meta["bad_control_example"],
        })
        results.append(QCResult(
            record=rec,
            qc_output=resp.content,
            overall_assessment=_parse_overall(resp.content),
        ))
    return {"qc_results": results}


def aggregate_results(state: GraphState) -> dict:
    """Aggregate QC results into a summary report."""
    cfg = _get_config(state)
    results = state["qc_results"]
    total = len(results)
    passed = sum(1 for r in results if r.overall_assessment == "MEETS")
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed / total:.0%}" if total else "N/A",
        "details": [
            {"control": r.record.control_description[:80], "assessment": r.overall_assessment}
            for r in results
        ],
    }
    out_dir = cfg.resolve_output_dir()

    if cfg.save_summary:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save full results: per-record QC output + metadata
    full_results = []
    for i, r in enumerate(results):
        entry = {
            "index": i,
            "risk_description": r.record.risk_description,
            "control_description": r.record.control_description,
            "overall_assessment": r.overall_assessment,
            "qc_output": r.qc_output,
        }
        full_results.append(entry)
    (out_dir / "full_results.json").write_text(json.dumps(full_results, indent=2), encoding="utf-8")

    return {"summary": summary}


def build_presentations(state: GraphState) -> dict:
    """Generate PPTX files for controls that did not fully meet standards."""
    cfg = _get_config(state)
    out_dir = cfg.resolve_output_dir()

    updated = []
    for i, r in enumerate(state["qc_results"]):
        if r.overall_assessment != "MEETS" and cfg.save_presentations:
            slug = re.sub(r"[^\w]+", "_", r.record.control_description[:40]).strip("_").lower()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"qc_report_{i}_{slug}_{ts}.pptx"
            build_qc_pptx(r.record.control_description, r.qc_output, path)
            r = r.model_copy(update={"pptx_path": str(path)})
        updated.append(r)
    return {"qc_results": updated}


def _route_after_aggregate(state: GraphState) -> str:
    """Route to PPT generation if any control failed, otherwise end."""
    cfg = _get_config(state)
    if cfg.save_presentations and any(
        r.overall_assessment != "MEETS" for r in state["qc_results"]
    ):
        return "build_presentations"
    return END


# ── Graph construction ───────────────────────────────────────────────────
def build_graph():
    """Build and compile the RCSA QC StateGraph."""
    g = StateGraph(GraphState)

    g.add_node("validate_input", validate_input)
    g.add_node("evaluate_controls", evaluate_controls)
    g.add_node("aggregate_results", aggregate_results)
    g.add_node("build_presentations", build_presentations)

    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "evaluate_controls")
    g.add_edge("evaluate_controls", "aggregate_results")
    g.add_conditional_edges(
        "aggregate_results",
        _route_after_aggregate,
        ["build_presentations", END],
    )
    g.add_edge("build_presentations", END)

    return g.compile()


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = RunConfig.load()

    app = build_graph()

    result = app.invoke({
        "config": config,
        "records": [
            
            {
                "risk_description": (
                    "a dog doesnt use a leash when walking in the park"
                ),
                "control_description": (
                    "a leash is used to restrain the dog when outside to prevent it from running away, getting lost, or causing harm to itself or others, ensuring safety and compliance with local regulations"
                ),
            },{
                "risk_description": (
                    "There is a risk that, in the absence of enforced role separation, "
                    "a single employee with unchecked access across transaction initiation, "
                    "approval, recording, and reconciliation functions could execute "
                    "unauthorized transactions, manipulate financial records, or conceal "
                    "fraudulent activity — resulting in direct financial loss, materially "
                    "inaccurate financial reporting, regulatory enforcement action, and "
                    "reputational harm to the institution"
                ),
                "control_description": (
                    "On an ongoing, continuous basis, the Access Management team in "
                    "coordination with the first-line Business Risk Officer enforces "
                    "role-based access controls (RBAC) to ensure that the initiation, "
                    "authorization, recording, and reconciliation of financial transactions "
                    "are assigned to separate and distinct individuals, with periodic "
                    "access reviews conducted no less than quarterly to validate continued "
                    "role segregation, using the institution's Identity and Access Management "
                    "(IAM) platform and core banking transaction processing system access "
                    "control modules to ensure no single employee retains end-to-end control "
                    "over a transaction lifecycle in a manner that could enable the execution "
                    "or concealment of unauthorized transactions or financial misstatement"
                ),
            }
        ],
    })

    print("\n=== Summary ===")
    print(json.dumps(result["summary"], indent=2))

    for r in result["qc_results"]:
        print(f"\n[{r.overall_assessment}] {r.record.control_description[:60]}...")
        if r.pptx_path:
            print(f"  PPTX saved: {r.pptx_path}")
