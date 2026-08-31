from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT / "agent/src", ROOT / "rag/src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from pcba_agent import AgentRunner
from pcba_agent.adapters.rag import AgentRAGAdapter
from pcba_agent.adapters.tools import ToolAdapter
from pcba_agent.config import load_settings
from pcba_rag.retriever import BgeM3QueryEncoder, Retriever, load_retriever_config


CLASSIFICATION_TOOL = "pcba_defect_classification"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecordingTools:
    def __init__(self, root: Path) -> None:
        self.delegate = ToolAdapter(root)
        self.case_id = ""
        self.calls: list[dict[str, Any]] = []
        self.failure_tool: str | None = None
        self.failure_used = False

    def begin_case(self, case_id: str, failure_tool: str | None = None) -> None:
        self.case_id = case_id
        self.calls = []
        self.failure_tool = failure_tool
        self.failure_used = False

    def _record(self, tool_name: str, success: bool, error: str | None = None) -> None:
        row: dict[str, Any] = {
            "case_id": self.case_id,
            "tool_name": tool_name,
            "success": success,
        }
        if error:
            row["error"] = error
        self.calls.append(row)

    def classify(self, image_path: str, request_id: str) -> dict[str, Any]:
        try:
            response = self.delegate.classify(image_path, request_id)
            self._record(CLASSIFICATION_TOOL, True)
            return response
        except Exception as exc:
            self._record(CLASSIFICATION_TOOL, False, str(exc))
            raise

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == self.failure_tool and not self.failure_used:
            self.failure_used = True
            message = f"T15 injected timeout for {tool_name}"
            self._record(tool_name, False, message)
            raise TimeoutError(message)
        try:
            response = self.delegate.invoke(tool_name, arguments)
            self._record(tool_name, True)
            return response
        except Exception as exc:
            self._record(tool_name, False, str(exc))
            raise


def make_retriever(qdrant_url: str, encoder: BgeM3QueryEncoder) -> Retriever:
    client = QdrantClient(url=qdrant_url, timeout=30, trust_env=False)
    retriever = Retriever(
        ROOT,
        client=client,
        encoder=encoder,
        check_connection=True,
    )
    retriever._owns_client = True
    return retriever


def tool_names(calls: list[dict[str, Any]]) -> list[str]:
    return [str(row["tool_name"]) for row in calls]


def multiset_match(expected: list[str], actual: list[str]) -> int:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    return sum(min(count, actual_counts[name]) for name, count in expected_counts.items())


def candidate_by_id(candidates: list[dict[str, Any]], relationship_id: str) -> dict[str, Any] | None:
    return next(
        (row for row in candidates if row.get("relationship_id") == relationship_id),
        None,
    )


def citation_counts(evidence: list[dict[str, Any]]) -> tuple[int, int]:
    complete = 0
    for row in evidence:
        citation = row.get("citation") or {}
        if (
            citation.get("source_id")
            and citation.get("pdf_page_start") is not None
            and citation.get("pdf_page_end") is not None
        ):
            complete += 1
    return complete, len(evidence)


def score_case(
    case: dict[str, Any],
    first: dict[str, Any],
    final: dict[str, Any],
    state: dict[str, Any],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    gold = case["gold_behavior"]
    expected_tools = list(gold.get("expected_tool_sequence", []))
    actual_tools = tool_names(calls)
    matched_calls = multiset_match(expected_tools, actual_tools)
    process_expected = [name for name in expected_tools if name != CLASSIFICATION_TOOL]
    process_actual = [name for name in actual_tools if name != CLASSIFICATION_TOOL]

    checks: dict[str, bool] = {
        "defect_correct": state.get("defect_name") == case["defect_gold"],
        "tool_sequence_exact": actual_tools == expected_tools,
        "process_tool_sequence_exact": process_actual == process_expected,
        "no_forbidden_tool": not bool(set(actual_tools) & set(gold.get("forbidden_tools", []))),
    }

    initial_expected = gold.get("first_turn_status") or gold.get("expected_first_status")
    if initial_expected:
        checks["first_status_correct"] = first.get("status") == initial_expected
    final_expected = (
        gold.get("expected_final_status_after_resume")
        or gold.get("expected_final_status")
        or gold.get("expected_first_status")
    )
    if final_expected:
        checks["final_status_correct"] = final.get("status") == final_expected

    expected_missing = gold.get("expected_missing_inputs_exact")
    if expected_missing is None:
        expected_missing = gold.get("first_turn_pending_inputs")
    if expected_missing is not None:
        checks["missing_inputs_exact"] = sorted(first.get("pending_inputs", [])) == sorted(
            expected_missing
        )

    expected_relationships = gold.get("expected_relationship_ids")
    if expected_relationships is not None:
        actual_relationships = [row.get("relationship_id") for row in final.get("candidates", [])]
        checks["candidate_paths_exact"] = set(actual_relationships) == set(expected_relationships)

    expected_status = gold.get("expected_assessment_status")
    if expected_status is not None:
        statuses = [row.get("assessment_status") for row in final.get("candidates", [])]
        checks["causal_status_correct"] = len(statuses) == 1 and statuses[0] == expected_status

    expected_candidate_statuses = gold.get("expected_candidate_statuses") or {}
    if expected_candidate_statuses:
        status_checks = []
        for relationship_id, expected in expected_candidate_statuses.items():
            candidate = candidate_by_id(final.get("candidates", []), relationship_id) or {}
            actual = (
                candidate.get("verification_capability")
                if expected == "unverified"
                else candidate.get("assessment_status")
            )
            status_checks.append(actual == expected)
        checks["causal_status_correct"] = all(status_checks)

    if gold.get("optimization_revalidation_required"):
        reflow_candidate = candidate_by_id(
            final.get("candidates", []), "REL-SHIFTED-COMPONENT-REFLOW"
        ) or {}
        optimization = reflow_candidate.get("optimization_result") or {}
        checks["optimization_revalidated"] = (
            process_actual.count("reflow_profile_prediction") >= 2
            and "reflow_parameter_optimization" in process_actual
            and optimization.get("recommendation_status") in {"accepted", "rejected"}
            and bool(optimization.get("revalidation_response"))
        )

    citation_complete, citation_total = citation_counts(final.get("rag_evidence", []))
    checks["rag_evidence_present"] = citation_total > 0
    checks["citations_complete"] = citation_total > 0 and citation_complete == citation_total

    return {
        "checks": checks,
        "strict_case_pass": all(checks.values()),
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "matched_tool_calls": matched_calls,
        "expected_tool_call_count": len(expected_tools),
        "actual_tool_call_count": len(actual_tools),
        "unnecessary_tool_call_count": len(actual_tools) - multiset_match(actual_tools, expected_tools),
        "expected_process_tools": process_expected,
        "actual_process_tools": process_actual,
        "citation_complete": citation_complete,
        "citation_total": citation_total,
    }


def request_from_case(case: dict[str, Any], run_id: str) -> dict[str, Any]:
    source = case["request"]
    image_path = ROOT / source["image_path"]
    return {
        "schema_version": "1.0.0",
        "request_id": f"{run_id}-{case['case_id']}",
        "thread_id": f"{run_id}-{case['case_id']}",
        "user_question": case["question"],
        "image_path": str(image_path.resolve()),
        "goal": source["goal"],
        "observations": source.get("observations", {}),
        "response_language": source.get("response_language", "zh"),
    }


def run_case(
    runner: AgentRunner,
    recorder: RecordingTools,
    case: dict[str, Any],
    run_id: str,
    failure_tool: str | None = None,
) -> dict[str, Any]:
    recorder.begin_case(case["case_id"], failure_tool=failure_tool)
    started = time.perf_counter()
    request = request_from_case(case, run_id)
    first_result = runner.invoke(request)
    first = first_result.model_dump(mode="json")
    final_result = first_result
    resume_used: dict[str, Any] | None = None
    gold = case["gold_behavior"]
    if first_result.status == "needs_input":
        explicit_resume = gold.get("resume")
        if explicit_resume:
            resume_used = explicit_resume
        elif gold.get("expected_final_status") == "completed":
            resume_used = {"unavailable_inputs": first_result.pending_inputs}
        if resume_used:
            final_result = runner.resume(request["thread_id"], resume_used)
    final = final_result.model_dump(mode="json")
    state = dict(runner.graph.get_state(runner._config(request["thread_id"])).values)
    score = score_case(case, first, final, state, list(recorder.calls))
    return {
        "case_id": case["case_id"],
        "duration_seconds": round(time.perf_counter() - started, 3),
        "resume_used": resume_used,
        "detected_defect": state.get("defect_name"),
        "defect_source": state.get("defect_source"),
        "external_tool_calls": list(recorder.calls),
        "score": score,
        "first_result": first,
        "final_result": final,
    }


def score_failure_variant(
    variant: dict[str, Any], case_result: dict[str, Any]
) -> dict[str, Any]:
    gold = variant["gold_behavior"]
    final = case_result["final_result"]
    calls = tool_names(case_result["external_tool_calls"])
    relationship_id = "REL-SHIFTED-COMPONENT-REFLOW"
    candidate = candidate_by_id(final.get("candidates", []), relationship_id) or {}
    response_text = final.get("response_text", "")
    checks = {
        "candidate_kept_inconclusive": candidate.get("assessment_status")
        == gold["candidate_assessment_status"],
        "tool_failure_recorded": bool(final.get("errors"))
        and any(not row.get("success") for row in case_result["external_tool_calls"]),
        "forbidden_followup_not_called": not bool(set(calls) & set(gold["must_not_call"])),
        "forbidden_claim_absent": not any(
            claim in response_text for claim in gold.get("must_not_claim", [])
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def aggregate(base_results: list[dict[str, Any]], variant_results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in base_results]
    expected_calls = sum(row["expected_tool_call_count"] for row in scores)
    actual_calls = sum(row["actual_tool_call_count"] for row in scores)
    matched_expected = sum(row["matched_tool_calls"] for row in scores)
    matched_actual = sum(
        row["actual_tool_call_count"] - row["unnecessary_tool_call_count"] for row in scores
    )
    process_expected = sum(len(row["expected_process_tools"]) for row in scores)
    process_matched = sum(
        multiset_match(row["expected_process_tools"], row["actual_process_tools"])
        for row in scores
    )
    unnecessary = sum(row["unnecessary_tool_call_count"] for row in scores)

    def rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else round(numerator / denominator, 6)

    missing_rows = [row for row in scores if "missing_inputs_exact" in row["checks"]]
    causal_rows = [row for row in scores if "causal_status_correct" in row["checks"]]
    revalidation_rows = [row for row in scores if "optimization_revalidated" in row["checks"]]
    citation_complete = sum(row["citation_complete"] for row in scores)
    citation_total = sum(row["citation_total"] for row in scores)
    classification_expected = sum(
        CLASSIFICATION_TOOL in row["expected_tools"] for row in scores
    )
    classification_actual = sum(
        CLASSIFICATION_TOOL in row["actual_tools"] for row in scores
    )

    return {
        "strict_case_pass_rate": rate(
            sum(row["strict_case_pass"] for row in scores), len(scores)
        ),
        "strict_cases_passed": sum(row["strict_case_pass"] for row in scores),
        "base_case_count": len(scores),
        "tool_call_recall": rate(matched_expected, expected_calls),
        "tool_call_precision": rate(matched_actual, actual_calls),
        "tool_call_f1": (
            None
            if not expected_calls or not actual_calls or not matched_expected
            else round(
                2
                * (matched_expected / expected_calls)
                * (matched_actual / actual_calls)
                / ((matched_expected / expected_calls) + (matched_actual / actual_calls)),
                6,
            )
        ),
        "unnecessary_tool_call_rate": rate(unnecessary, actual_calls),
        "exact_tool_sequence_rate": rate(
            sum(row["checks"]["tool_sequence_exact"] for row in scores), len(scores)
        ),
        "process_tool_recall_excluding_aoi": rate(process_matched, process_expected),
        "process_tool_exact_sequence_rate": rate(
            sum(row["checks"]["process_tool_sequence_exact"] for row in scores),
            len(scores),
        ),
        "aoi_classification_call_recall": rate(classification_actual, classification_expected),
        "defect_identification_accuracy": rate(
            sum(row["checks"]["defect_correct"] for row in scores), len(scores)
        ),
        "missing_input_exact_match_rate": rate(
            sum(row["checks"]["missing_inputs_exact"] for row in missing_rows),
            len(missing_rows),
        ),
        "candidate_causal_status_accuracy": rate(
            sum(row["checks"]["causal_status_correct"] for row in causal_rows),
            len(causal_rows),
        ),
        "optimization_revalidation_rate": rate(
            sum(row["checks"]["optimization_revalidated"] for row in revalidation_rows),
            len(revalidation_rows),
        ),
        "citation_completeness": rate(citation_complete, citation_total),
        "rag_evidence_returned": citation_total,
        "tool_failure_safe_degradation_rate": rate(
            sum(row["score"]["pass"] for row in variant_results), len(variant_results)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen T15 Agent system evaluation")
    parser.add_argument(
        "--dataset",
        default="agent/evaluation/dataset/t15_mvp_cases.v0.1.json",
    )
    parser.add_argument(
        "--manifest",
        default="agent/evaluation/dataset/FROZEN_MANIFEST.json",
    )
    parser.add_argument(
        "--output",
        default="agent/evaluation/results/t15_evaluation_results.v0.1.json",
    )
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16333")
    args = parser.parse_args()

    dataset_path = (ROOT / args.dataset).resolve()
    manifest_path = (ROOT / args.manifest).resolve()
    output_path = (ROOT / args.output).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = next(
        row["sha256"]
        for row in manifest["files"]
        if row["role"] == "gold_dataset"
    )
    actual_hash = sha256_file(dataset_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Frozen dataset hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    run_id = f"t15-{int(time.time())}"
    started_at = utc_now()
    settings = load_settings()
    recorder = RecordingTools(ROOT)
    shared_query_encoder = BgeM3QueryEncoder(ROOT, load_retriever_config(ROOT))
    rag = AgentRAGAdapter(
        ROOT,
        settings.raw["rag"],
        retriever_factory=lambda: make_retriever(
            args.qdrant_url,
            shared_query_encoder,
        ),
    )
    base_results: list[dict[str, Any]] = []
    variant_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pcba_t15_") as temp_dir:
        checkpoint = Path(temp_dir) / "checkpoints.sqlite3"
        with AgentRunner(
            settings,
            rag=rag,
            tools=recorder,
            checkpoint_path=checkpoint,
        ) as runner:
            cases_by_id = {row["case_id"]: row for row in dataset["cases"]}
            for case in dataset["cases"]:
                print(f"START base case {case['case_id']}", flush=True)
                result = run_case(runner, recorder, case, run_id)
                base_results.append(result)
                print(
                    f"DONE base case {case['case_id']} "
                    f"in {result['duration_seconds']}s",
                    flush=True,
                )
            for variant in dataset["execution_variants"]:
                source_case = cases_by_id[variant["source_case_id"]]
                variant_case = json.loads(json.dumps(source_case, ensure_ascii=False))
                variant_case["case_id"] = variant["variant_id"]
                print(f"START failure variant {variant['variant_id']}", flush=True)
                result = run_case(
                    runner,
                    recorder,
                    variant_case,
                    run_id,
                    failure_tool=variant["injection"]["tool_name"],
                )
                result["score"] = score_failure_variant(variant, result)
                variant_results.append(result)
                print(
                    f"DONE failure variant {variant['variant_id']} "
                    f"in {result['duration_seconds']}s",
                    flush=True,
                )

    report = {
        "schema_version": "1.0.0",
        "evaluation_id": run_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "dataset": {
            "path": dataset_path.relative_to(ROOT).as_posix(),
            "sha256": actual_hash,
            "dataset_id": dataset["dataset_id"],
            "review_status": dataset["review_status"],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "qdrant_url": args.qdrant_url,
            "qwen_model": settings.raw["llm"]["model"],
            "rag_reranker_weight": settings.raw["rag"]["reranker_weight"],
        },
        "metrics": aggregate(base_results, variant_results),
        "base_results": base_results,
        "variant_results": variant_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"RESULT={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
