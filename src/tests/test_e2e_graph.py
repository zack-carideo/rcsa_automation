"""Unit tests for src/chains/e2e_graph.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path when running tests directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.chains.e2e_graph import (
    _GROUNDING_THRESHOLD,
    ControlRecord,
    GraphState,
    GroundingResult,
    QCResult,
    _compute_coverage,
    _extract_key_terms,
    _parse_overall,
    aggregate_results,
    build_graph,
    check_grounding,
    validate_input,
    _route_after_aggregate,
)
from src.configs.config import RunConfig
from langgraph.graph import END


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_record() -> ControlRecord:
    return ControlRecord(
        risk_description="unauthorized access to sensitive financial data",
        control_description="access management team reviews permissions quarterly using IAM platform",
    )


@pytest.fixture
def grounded_qc_result(sample_record: ControlRecord) -> QCResult:
    # qc_output contains key terms from both risk and control descriptions
    return QCResult(
        record=sample_record,
        qc_output=(
            "The access management team reviews permissions using the IAM platform. "
            "Unauthorized access to sensitive financial data is mitigated quarterly. "
            "| Overall Assessment | MEETS | Adequate controls in place |"
        ),
        overall_assessment="MEETS",
    )


@pytest.fixture
def ungrounded_qc_result(sample_record: ControlRecord) -> QCResult:
    # qc_output is generic — shares almost no terms with the input
    return QCResult(
        record=sample_record,
        qc_output="This is a generic placeholder response with no specific content.",
        overall_assessment="DOES NOT MEET",
    )


def _make_grounding(is_grounded: bool, risk_cov: float = 0.8, ctrl_cov: float = 0.8) -> GroundingResult:
    return GroundingResult(
        is_grounded=is_grounded,
        risk_term_coverage=risk_cov,
        control_term_coverage=ctrl_cov,
        threshold=_GROUNDING_THRESHOLD,
    )


# ── _parse_overall ─────────────────────────────────────────────────────────

class TestParseOverall:
    def test_meets(self):
        text = "| Overall Assessment | MEETS | All criteria satisfied |"
        assert _parse_overall(text) == "MEETS"

    def test_partially_meets(self):
        text = "| Overall Assessment | PARTIALLY MEETS | Some gaps |"
        assert _parse_overall(text) == "PARTIALLY MEETS"

    def test_does_not_meet(self):
        text = "| Overall Assessment | DOES NOT MEET | Missing elements |"
        assert _parse_overall(text) == "DOES NOT MEET"

    def test_case_insensitive(self):
        text = "| Overall Assessment | meets | lowercase rating |"
        assert _parse_overall(text) == "MEETS"

    def test_missing_returns_does_not_meet(self):
        assert _parse_overall("No assessment here at all.") == "DOES NOT MEET"

    def test_whitespace_variants(self):
        text = "| Overall  Assessment |  PARTIALLY MEETS  | ok |"
        assert _parse_overall(text) == "PARTIALLY MEETS"


# ── _extract_key_terms ────────────────────────────────────────────────────

class TestExtractKeyTerms:
    def test_returns_lowercase(self):
        terms = _extract_key_terms("FINANCIAL Transaction Controls")
        assert all(t == t.lower() for t in terms)

    def test_filters_stop_words(self):
        terms = _extract_key_terms("the and with from that")
        assert terms == []

    def test_filters_short_words(self):
        # Words of 3 chars or fewer are excluded
        terms = _extract_key_terms("a an the cat dog at is")
        assert not any(len(t) <= 3 for t in terms)

    def test_deduplicates(self):
        terms = _extract_key_terms("access access access permissions permissions")
        assert terms.count("access") == 1
        assert terms.count("permissions") == 1

    def test_empty_string(self):
        assert _extract_key_terms("") == []

    def test_extracts_meaningful_terms(self):
        terms = _extract_key_terms("access management quarterly review")
        assert "access" in terms
        assert "management" in terms
        assert "quarterly" in terms
        assert "review" in terms

    def test_preserves_order_of_first_occurrence(self):
        terms = _extract_key_terms("alpha beta alpha gamma beta")
        assert terms == ["alpha", "beta", "gamma"]


# ── _compute_coverage ─────────────────────────────────────────────────────

class TestComputeCoverage:
    def test_all_found(self):
        coverage, missing = _compute_coverage(["access", "review"], "access and review procedures")
        assert coverage == 1.0
        assert missing == []

    def test_none_found(self):
        coverage, missing = _compute_coverage(["access", "review"], "unrelated generic text here")
        assert coverage == 0.0
        assert set(missing) == {"access", "review"}

    def test_partial_coverage(self):
        coverage, missing = _compute_coverage(["access", "review", "quarterly"], "access the records")
        assert coverage == pytest.approx(1 / 3)
        assert "review" in missing
        assert "quarterly" in missing

    def test_empty_terms_returns_full_coverage(self):
        coverage, missing = _compute_coverage([], "some output text")
        assert coverage == 1.0
        assert missing == []

    def test_case_insensitive_matching(self):
        coverage, missing = _compute_coverage(["access"], "ACCESS controls are reviewed")
        assert coverage == 1.0
        assert missing == []


# ── check_grounding ───────────────────────────────────────────────────────

class TestCheckGrounding:
    def test_grounded_result_passes(self, grounded_qc_result: QCResult):
        state: GraphState = {"qc_results": [grounded_qc_result]}
        result = check_grounding(state)
        qc = result["qc_results"][0]
        assert qc.grounding is not None
        assert qc.grounding.is_grounded is True

    def test_ungrounded_result_fails(self, ungrounded_qc_result: QCResult):
        state: GraphState = {"qc_results": [ungrounded_qc_result]}
        result = check_grounding(state)
        qc = result["qc_results"][0]
        assert qc.grounding is not None
        assert qc.grounding.is_grounded is False

    def test_grounding_field_populated(self, grounded_qc_result: QCResult):
        state: GraphState = {"qc_results": [grounded_qc_result]}
        result = check_grounding(state)
        qc = result["qc_results"][0]
        g = qc.grounding
        assert g is not None
        assert 0.0 <= g.risk_term_coverage <= 1.0
        assert 0.0 <= g.control_term_coverage <= 1.0
        assert g.threshold == _GROUNDING_THRESHOLD

    def test_multiple_records_all_checked(self, sample_record: ControlRecord):
        results = [
            QCResult(
                record=sample_record,
                qc_output="access management quarterly review IAM permissions financial",
                overall_assessment="MEETS",
            ),
            QCResult(
                record=sample_record,
                qc_output="irrelevant placeholder output text here",
                overall_assessment="DOES NOT MEET",
            ),
        ]
        state: GraphState = {"qc_results": results}
        out = check_grounding(state)
        assert len(out["qc_results"]) == 2
        assert all(r.grounding is not None for r in out["qc_results"])

    def test_missing_risk_terms_capped_at_ten(self, sample_record: ControlRecord):
        # Build a record with many unique key terms in the risk description
        long_risk = " ".join(f"termword{i}" for i in range(20))
        record = ControlRecord(risk_description=long_risk, control_description="control action")
        qc = QCResult(record=record, qc_output="unrelated output", overall_assessment="DOES NOT MEET")
        state: GraphState = {"qc_results": [qc]}
        out = check_grounding(state)
        assert len(out["qc_results"][0].grounding.missing_risk_terms) <= 10

    def test_preserves_other_qc_result_fields(self, grounded_qc_result: QCResult):
        state: GraphState = {"qc_results": [grounded_qc_result]}
        out = check_grounding(state)
        qc = out["qc_results"][0]
        assert qc.record == grounded_qc_result.record
        assert qc.qc_output == grounded_qc_result.qc_output
        assert qc.overall_assessment == grounded_qc_result.overall_assessment


# ── validate_input ────────────────────────────────────────────────────────

class TestValidateInput:
    def test_valid_dicts_are_converted(self):
        state: GraphState = {
            "records": [
                {"risk_description": "some risk", "control_description": "some control"},
            ]
        }
        result = validate_input(state)
        assert len(result["records"]) == 1
        assert isinstance(result["records"][0], ControlRecord)

    def test_valid_controlrecord_objects_pass_through(self):
        rec = ControlRecord(risk_description="risk", control_description="control")
        state: GraphState = {"records": [rec]}
        result = validate_input(state)
        assert result["records"][0] == rec

    def test_empty_risk_description_raises(self):
        from pydantic import ValidationError
        state: GraphState = {
            "records": [{"risk_description": "", "control_description": "valid"}]
        }
        with pytest.raises(ValidationError):
            validate_input(state)

    def test_empty_control_description_raises(self):
        from pydantic import ValidationError
        state: GraphState = {
            "records": [{"risk_description": "valid", "control_description": ""}]
        }
        with pytest.raises(ValidationError):
            validate_input(state)

    def test_multiple_valid_records(self):
        state: GraphState = {
            "records": [
                {"risk_description": "risk one", "control_description": "control one"},
                {"risk_description": "risk two", "control_description": "control two"},
            ]
        }
        result = validate_input(state)
        assert len(result["records"]) == 2


# ── aggregate_results ─────────────────────────────────────────────────────

class TestAggregateResults:
    def _make_state(self, tmp_path: Path, qc_results: list) -> GraphState:
        cfg = RunConfig(output_dir=tmp_path, save_summary=True, save_presentations=False)
        return {"config": cfg, "qc_results": qc_results}

    def _make_qc(self, assessment: str, grounded: bool, record: ControlRecord) -> QCResult:
        return QCResult(
            record=record,
            qc_output="some output",
            overall_assessment=assessment,
            grounding=_make_grounding(grounded),
        )

    def test_pass_fail_counts(self, tmp_path: Path, sample_record: ControlRecord):
        qcs = [
            self._make_qc("MEETS", True, sample_record),
            self._make_qc("DOES NOT MEET", False, sample_record),
            self._make_qc("PARTIALLY MEETS", True, sample_record),
        ]
        result = aggregate_results(self._make_state(tmp_path, qcs))
        summary = result["summary"]
        assert summary["total"] == 3
        assert summary["passed"] == 1
        assert summary["failed"] == 2

    def test_grounding_failure_count(self, tmp_path: Path, sample_record: ControlRecord):
        qcs = [
            self._make_qc("MEETS", True, sample_record),
            self._make_qc("DOES NOT MEET", False, sample_record),
        ]
        result = aggregate_results(self._make_state(tmp_path, qcs))
        assert result["summary"]["grounding_failures"] == 1

    def test_pass_rate_format(self, tmp_path: Path, sample_record: ControlRecord):
        qcs = [
            self._make_qc("MEETS", True, sample_record),
            self._make_qc("MEETS", True, sample_record),
            self._make_qc("DOES NOT MEET", False, sample_record),
            self._make_qc("DOES NOT MEET", False, sample_record),
        ]
        result = aggregate_results(self._make_state(tmp_path, qcs))
        assert result["summary"]["pass_rate"] == "50%"

    def test_empty_results_pass_rate(self, tmp_path: Path):
        result = aggregate_results(self._make_state(tmp_path, []))
        assert result["summary"]["pass_rate"] == "N/A"

    def test_summary_json_written(self, tmp_path: Path, sample_record: ControlRecord):
        qcs = [self._make_qc("MEETS", True, sample_record)]
        aggregate_results(self._make_state(tmp_path, qcs))
        summary_file = tmp_path / "summary.json"
        assert summary_file.exists()
        data = json.loads(summary_file.read_text())
        assert data["total"] == 1

    def test_full_results_json_written(self, tmp_path: Path, sample_record: ControlRecord):
        qcs = [self._make_qc("MEETS", True, sample_record)]
        aggregate_results(self._make_state(tmp_path, qcs))
        full_file = tmp_path / "full_results.json"
        assert full_file.exists()
        data = json.loads(full_file.read_text())
        assert len(data) == 1
        assert data[0]["grounding"] is not None

    def test_full_results_includes_grounding_data(self, tmp_path: Path, sample_record: ControlRecord):
        qcs = [self._make_qc("DOES NOT MEET", False, sample_record)]
        aggregate_results(self._make_state(tmp_path, qcs))
        data = json.loads((tmp_path / "full_results.json").read_text())
        grounding = data[0]["grounding"]
        assert "is_grounded" in grounding
        assert grounding["is_grounded"] is False

    def test_details_include_grounded_flag(self, tmp_path: Path, sample_record: ControlRecord):
        qcs = [self._make_qc("MEETS", True, sample_record)]
        result = aggregate_results(self._make_state(tmp_path, qcs))
        detail = result["summary"]["details"][0]
        assert "grounded" in detail
        assert detail["grounded"] is True


# ── _route_after_aggregate ────────────────────────────────────────────────

class TestRouteAfterAggregate:
    def _make_state(self, assessments: list[str], save_presentations: bool) -> GraphState:
        cfg = RunConfig(save_presentations=save_presentations)
        records = [
            QCResult(
                record=ControlRecord(risk_description="risk", control_description="control"),
                qc_output="output",
                overall_assessment=a,
            )
            for a in assessments
        ]
        return {"config": cfg, "qc_results": records}

    def test_routes_to_build_presentations_when_failures_exist(self):
        state = self._make_state(["MEETS", "DOES NOT MEET"], save_presentations=True)
        assert _route_after_aggregate(state) == "build_presentations"

    def test_routes_to_end_when_all_pass(self):
        state = self._make_state(["MEETS", "MEETS"], save_presentations=True)
        assert _route_after_aggregate(state) == END

    def test_routes_to_end_when_presentations_disabled(self):
        state = self._make_state(["DOES NOT MEET"], save_presentations=False)
        assert _route_after_aggregate(state) == END

    def test_routes_to_end_on_empty_results(self):
        state = self._make_state([], save_presentations=True)
        assert _route_after_aggregate(state) == END


# ── build_graph ───────────────────────────────────────────────────────────

class TestBuildGraph:
    def test_graph_compiles_without_error(self):
        app = build_graph()
        assert app is not None

    def test_graph_has_expected_nodes(self):
        app = build_graph()
        node_names = set(app.nodes.keys())
        expected = {"validate_input", "evaluate_controls", "check_grounding",
                    "aggregate_results", "build_presentations"}
        assert expected.issubset(node_names)
