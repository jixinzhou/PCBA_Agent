from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .full_page_pipeline import _atomic_json
from .sample_pipeline import collect_environment


_PROCESS_ORDER = ("printing", "placement", "reflow")
_DEFECT_ORDER = (
    "insufficient_solder",
    "excessive_solder",
    "short",
    "shifted_component",
)
_EVIDENCE_ORDER = (
    "defect_mechanism",
    "troubleshooting_guidance",
    "process_guideline",
    "normative_requirement",
)
_LATIN_ALNUM = re.compile(r"[a-z0-9]")


@dataclass(frozen=True)
class TermRule:
    rule_id: str
    category: str
    term: str
    process_ids: tuple[str, ...] = ()
    defect_ids: tuple[str, ...] = ()
    evidence_roles: tuple[str, ...] = ()
    source_entity_id: str | None = None
    relationship_id: str | None = None


@dataclass(frozen=True)
class MappingContext:
    config: dict[str, Any]
    dictionary: dict[str, Any]
    ontology: dict[str, Any]
    term_rules: tuple[TermRule, ...]
    process_entities: dict[str, dict[str, str]]
    defect_entities: dict[str, dict[str, str]]
    disabled_terms: tuple[str, ...]


def load_mapping_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "rag/config/metadata_mapping.v0.1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_metadata_validators(
    project_root: Path,
) -> tuple[Draft202012Validator, Draft202012Validator]:
    metadata_schema = json.loads(
        (project_root / "rag/schemas/metadata.v1.1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    chunk_schema = json.loads(
        (project_root / "rag/schemas/chunk.v1.1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    trace_schema = json.loads(
        (project_root / "rag/schemas/mapping_trace.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for schema in (metadata_schema, chunk_schema, trace_schema):
        Draft202012Validator.check_schema(schema)
    registry = Registry().with_resource(
        "metadata.v1.1.schema.json", Resource.from_contents(metadata_schema)
    )
    return (
        Draft202012Validator(chunk_schema, registry=registry),
        Draft202012Validator(trace_schema),
    )


def normalize_for_match(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def contains_term(value: str, term: str) -> bool:
    if not value or not term:
        return False
    if _LATIN_ALNUM.search(term):
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        return re.search(pattern, value) is not None
    return term in value


def _ordered(values: set[str], order: tuple[str, ...]) -> list[str]:
    unknown = values - set(order)
    if unknown:
        raise ValueError(f"Unknown values for ordered output: {sorted(unknown)}")
    return [value for value in order if value in values]


def _entity_terms(entity: dict[str, Any]) -> list[str]:
    values = [
        entity["canonical_name"].replace("_", " "),
        entity["display_name_zh"],
        *entity.get("aliases", []),
    ]
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_for_match(str(value))
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(str(value))
    return result


def _entity_ref(entity: dict[str, Any]) -> dict[str, str]:
    return {
        "entity_id": entity["entity_id"],
        "canonical_name": entity["canonical_name"],
        "display_name_zh": entity["display_name_zh"],
    }


def _validate_authorities(
    config: dict[str, Any], dictionary: dict[str, Any], ontology: dict[str, Any]
) -> None:
    if dictionary["ontology_version"] != ontology["ontology_version"]:
        raise ValueError("Entity dictionary and ontology versions differ")
    if tuple(ontology["scope"]["process_ids"]) != _PROCESS_ORDER:
        raise ValueError("Ontology process scope differs from Metadata V1.1")
    if tuple(ontology["scope"]["defect_ids"]) != _DEFECT_ORDER:
        raise ValueError("Ontology defect scope differs from Metadata V1.1")
    if config["dictionary_path"] != "rag/schemas/entity_dictionary.json":
        raise ValueError("Unexpected entity dictionary authority")
    if config["ontology_path"] != "ontology/pcba_defect_causality.v1.1.yaml":
        raise ValueError("Unexpected ontology authority")

    dictionary_entities = {
        entity["entity_id"]: entity for entity in dictionary["entities"]
    }
    for group in ("processes", "defects", "candidate_causes"):
        for entity in ontology[group]:
            counterpart = dictionary_entities.get(entity["entity_id"])
            if counterpart is None:
                raise ValueError(f"Dictionary missing {entity['entity_id']}")
            for field in ("canonical_name", "display_name_zh"):
                if counterpart[field] != entity[field]:
                    raise ValueError(
                        f"Dictionary mismatch for {entity['entity_id']} field {field}"
                    )


def _make_rule(
    *,
    rule_id: str,
    category: str,
    term: str,
    process_ids: list[str] | tuple[str, ...] = (),
    defect_ids: list[str] | tuple[str, ...] = (),
    evidence_roles: list[str] | tuple[str, ...] = (),
    source_entity_id: str | None = None,
    relationship_id: str | None = None,
) -> TermRule:
    return TermRule(
        rule_id=rule_id,
        category=category,
        term=normalize_for_match(term),
        process_ids=tuple(process_ids),
        defect_ids=tuple(defect_ids),
        evidence_roles=tuple(evidence_roles),
        source_entity_id=source_entity_id,
        relationship_id=relationship_id,
    )


def _build_term_rules(
    config: dict[str, Any], ontology: dict[str, Any]
) -> tuple[TermRule, ...]:
    disabled = {
        normalize_for_match(term)
        for term in config["matching"].get("disabled_bare_terms", [])
    }
    rules: list[TermRule] = []

    for entity in ontology["processes"]:
        for term in _entity_terms(entity):
            rule = _make_rule(
                rule_id=f"ontology-process:{entity['entity_id']}",
                category="process_term",
                term=term,
                process_ids=[entity["canonical_name"]],
                source_entity_id=entity["entity_id"],
            )
            if rule.term not in disabled:
                rules.append(rule)

    for entity in ontology["defects"]:
        for term in _entity_terms(entity):
            rule = _make_rule(
                rule_id=f"ontology-defect:{entity['entity_id']}",
                category="synonym",
                term=term,
                defect_ids=[entity["canonical_name"]],
                source_entity_id=entity["entity_id"],
            )
            if rule.term not in disabled:
                rules.append(rule)

    relations_by_cause = {
        relation["candidate_cause_id"]: relation
        for relation in ontology["relationships"]
    }
    for entity in ontology["candidate_causes"]:
        relation = relations_by_cause[entity["entity_id"]]
        for term in _entity_terms(entity):
            rule = _make_rule(
                rule_id=f"ontology-cause:{entity['entity_id']}",
                category="candidate_cause",
                term=term,
                process_ids=[relation["process_id"]],
                defect_ids=[relation["defect_id"]],
                source_entity_id=entity["entity_id"],
                relationship_id=relation["relationship_id"],
            )
            if rule.term not in disabled:
                rules.append(rule)

    for item in config.get("related_terms", []):
        for term in item["terms"]:
            rules.append(
                _make_rule(
                    rule_id=item["rule_id"],
                    category="related_term",
                    term=term,
                    process_ids=item.get("process_ids", []),
                    defect_ids=item.get("defect_ids", []),
                )
            )

    for item in config.get("process_terms", []):
        for term in item["terms"]:
            rules.append(
                _make_rule(
                    rule_id=item["rule_id"],
                    category="process_term",
                    term=term,
                    process_ids=[item["process_id"]],
                )
            )

    for item in config.get("evidence_role_rules", []):
        for term in item["terms"]:
            rules.append(
                _make_rule(
                    rule_id=item["rule_id"],
                    category="evidence_role_rule",
                    term=term,
                    evidence_roles=[item["evidence_role"]],
                )
            )

    deduplicated: dict[tuple[Any, ...], TermRule] = {}
    for rule in rules:
        if not rule.term or rule.term in disabled:
            continue
        key = (
            rule.category,
            rule.term,
            rule.process_ids,
            rule.defect_ids,
            rule.evidence_roles,
            rule.source_entity_id,
            rule.relationship_id,
        )
        deduplicated.setdefault(key, rule)
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda value: (-len(value.term), value.category, value.term),
        )
    )


def build_mapping_context(project_root: Path) -> MappingContext:
    config = load_mapping_config(project_root)
    dictionary = json.loads(
        (project_root / config["dictionary_path"]).read_text(encoding="utf-8")
    )
    ontology = yaml.safe_load(
        (project_root / config["ontology_path"]).read_text(encoding="utf-8")
    )
    _validate_authorities(config, dictionary, ontology)
    processes = {
        entity["canonical_name"]: _entity_ref(entity)
        for entity in ontology["processes"]
    }
    defects = {
        entity["canonical_name"]: _entity_ref(entity)
        for entity in ontology["defects"]
    }
    disabled = tuple(
        normalize_for_match(term)
        for term in config["matching"].get("disabled_bare_terms", [])
    )
    return MappingContext(
        config=config,
        dictionary=dictionary,
        ontology=ontology,
        term_rules=_build_term_rules(config, ontology),
        process_entities=processes,
        defect_entities=defects,
        disabled_terms=disabled,
    )


def _excluded_section(
    section_path: list[str], config: dict[str, Any]
) -> tuple[bool, str | None]:
    exact = {
        normalize_for_match(value)
        for value in config.get("excluded_sections", {}).get("exact", [])
    }
    for section in section_path:
        normalized = normalize_for_match(section)
        if normalized in exact:
            return True, f"excluded-section:{section}"
    return False, None


def _trace_match(rule: TermRule, matched_in: str) -> dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "term_category": rule.category,
        "matched_term": rule.term,
        "matched_in": matched_in,
        "source_entity_id": rule.source_entity_id,
        "relationship_id": rule.relationship_id,
        "assigned_process_ids": list(rule.process_ids),
        "assigned_defect_ids": list(rule.defect_ids),
        "assigned_evidence_roles": list(rule.evidence_roles),
    }


def _curated_match(
    rule_id: str,
    matched_term: str,
    matched_in: str,
    process_ids: list[str] | None = None,
    defect_ids: list[str] | None = None,
    evidence_roles: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "term_category": "curated_default",
        "matched_term": matched_term,
        "matched_in": matched_in,
        "source_entity_id": None,
        "relationship_id": None,
        "assigned_process_ids": process_ids or [],
        "assigned_defect_ids": defect_ids or [],
        "assigned_evidence_roles": evidence_roles or [],
    }


def validate_metadata_invariants(
    source: dict[str, Any], enriched: dict[str, Any], trace: dict[str, Any]
) -> None:
    immutable_fields = (
        "chunk_id",
        "source_id",
        "page_ids",
        "pdf_page_start",
        "pdf_page_end",
        "section_path",
        "text",
        "text_hash",
    )
    for field in immutable_fields:
        if enriched[field] != source[field]:
            raise ValueError(f"T10.5 changed immutable Chunk field {field}")
    metadata = enriched["metadata"]
    if [item["canonical_name"] for item in metadata["process_entities"]] != metadata[
        "process_ids"
    ]:
        raise ValueError("process_entities do not match process_ids")
    if [item["canonical_name"] for item in metadata["defect_entities"]] != metadata[
        "defect_ids"
    ]:
        raise ValueError("defect_entities do not match defect_ids")
    if metadata["process_ids"] != trace["final_process_ids"]:
        raise ValueError("Trace process assignments differ from Metadata")
    if metadata["defect_ids"] != trace["final_defect_ids"]:
        raise ValueError("Trace defect assignments differ from Metadata")
    if metadata["evidence_roles"] != trace["final_evidence_roles"]:
        raise ValueError("Trace evidence assignments differ from Metadata")
    if trace["excluded_from_semantic_tagging"] and any(
        (metadata["process_ids"], metadata["defect_ids"], metadata["evidence_roles"])
    ):
        raise ValueError("Excluded structural Chunk received semantic tags")


def map_chunk(
    chunk: dict[str, Any], context: MappingContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = context.config
    excluded, exclusion_rule = _excluded_section(chunk["section_path"], config)
    section_text = normalize_for_match(" > ".join(chunk["section_path"]))
    body_text = normalize_for_match(chunk["text"])
    matches: list[dict[str, Any]] = []

    if not excluded:
        for rule in context.term_rules:
            matched_in = None
            if contains_term(section_text, rule.term):
                matched_in = "section_path"
            elif contains_term(body_text, rule.term):
                matched_in = "text"
            if matched_in:
                matches.append(_trace_match(rule, matched_in))

        defaults = config.get("source_defaults", {}).get(chunk["source_id"], {})
        if defaults and any(
            defaults.get(field)
            for field in ("process_ids", "defect_ids", "evidence_roles")
        ):
            matches.append(
                _curated_match(
                    rule_id=f"source-default:{chunk['source_id']}",
                    matched_term=chunk["source_id"],
                    matched_in="source_default",
                    process_ids=defaults.get("process_ids"),
                    defect_ids=defaults.get("defect_ids"),
                    evidence_roles=defaults.get("evidence_roles"),
                )
            )

        for item in config.get("section_rules", []):
            if item["source_id"] != chunk["source_id"]:
                continue
            section_term = normalize_for_match(item["section_contains"])
            if contains_term(section_text, section_term):
                matches.append(
                    _curated_match(
                        rule_id=item["rule_id"],
                        matched_term=section_term,
                        matched_in="section_path",
                        process_ids=item.get("process_ids"),
                        defect_ids=item.get("defect_ids"),
                        evidence_roles=item.get("evidence_roles"),
                    )
                )

    process_ids = _ordered(
        {
            process_id
            for match in matches
            for process_id in match["assigned_process_ids"]
        },
        _PROCESS_ORDER,
    )
    defect_ids = _ordered(
        {
            defect_id
            for match in matches
            for defect_id in match["assigned_defect_ids"]
        },
        _DEFECT_ORDER,
    )
    evidence_roles = _ordered(
        {
            role for match in matches for role in match["assigned_evidence_roles"]
        },
        _EVIDENCE_ORDER,
    )
    origins = {
        "curated" if match["term_category"] == "curated_default" else "rule"
        for match in matches
    }
    if origins == {"rule", "curated"}:
        tag_origin = "mixed"
    elif origins == {"curated"}:
        tag_origin = "curated"
    elif origins == {"rule"}:
        tag_origin = "rule"
    else:
        tag_origin = "none"

    old_metadata = chunk["metadata"]
    metadata = {
        "schema_version": "1.1.0",
        "source_title": old_metadata["source_title"],
        "organization": old_metadata["organization"],
        "document_type": old_metadata["document_type"],
        "language": old_metadata["language"],
        "rights_status": old_metadata["rights_status"],
        "dictionary_version": context.dictionary["dictionary_version"],
        "ontology_version": context.ontology["ontology_version"],
        "mapping_version": config["mapping_version"],
        "process_ids": process_ids,
        "process_entities": [context.process_entities[value] for value in process_ids],
        "defect_ids": defect_ids,
        "defect_entities": [context.defect_entities[value] for value in defect_ids],
        "evidence_roles": evidence_roles,
        "tag_origin": tag_origin,
        "parser_version": old_metadata["parser_version"],
        "mapper_version": config["mapper_version"],
        "ocr_used": old_metadata["ocr_used"],
    }
    enriched = {**chunk, "schema_version": "1.1.0", "metadata": metadata}
    trace = {
        "schema_version": "1.0.0",
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "mapping_version": config["mapping_version"],
        "dictionary_version": context.dictionary["dictionary_version"],
        "ontology_version": context.ontology["ontology_version"],
        "excluded_from_semantic_tagging": excluded,
        "exclusion_rule": exclusion_rule,
        "matches": matches,
        "final_process_ids": process_ids,
        "final_defect_ids": defect_ids,
        "final_evidence_roles": evidence_roles,
    }
    validate_metadata_invariants(chunk, enriched, trace)
    return enriched, trace


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
    return records


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _source_summary(
    source_id: str,
    records: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    output_path: Path,
    trace_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    process_counts = Counter(
        value for record in records for value in record["metadata"]["process_ids"]
    )
    defect_counts = Counter(
        value for record in records for value in record["metadata"]["defect_ids"]
    )
    evidence_counts = Counter(
        value for record in records for value in record["metadata"]["evidence_roles"]
    )
    category_counts = Counter(
        match["term_category"] for trace in traces for match in trace["matches"]
    )
    return {
        "source_id": source_id,
        "chunk_count": len(records),
        "schema_valid_chunks": len(records),
        "trace_count": len(traces),
        "excluded_structural_chunks": sum(
            trace["excluded_from_semantic_tagging"] for trace in traces
        ),
        "chunks_with_process": sum(bool(r["metadata"]["process_ids"]) for r in records),
        "chunks_with_defect": sum(bool(r["metadata"]["defect_ids"]) for r in records),
        "chunks_with_evidence_role": sum(
            bool(r["metadata"]["evidence_roles"]) for r in records
        ),
        "multi_process_chunks": sum(len(r["metadata"]["process_ids"]) > 1 for r in records),
        "multi_defect_chunks": sum(len(r["metadata"]["defect_ids"]) > 1 for r in records),
        "process_counts": {key: process_counts.get(key, 0) for key in _PROCESS_ORDER},
        "defect_counts": {key: defect_counts.get(key, 0) for key in _DEFECT_ORDER},
        "evidence_role_counts": {
            key: evidence_counts.get(key, 0) for key in _EVIDENCE_ORDER
        },
        "match_category_counts": dict(sorted(category_counts.items())),
        "output_path": str(output_path.relative_to(project_root)),
        "trace_path": str(trace_path.relative_to(project_root)),
    }


def render_metadata_report(summary: dict[str, Any]) -> str:
    lines = [
        "# T10.5 Chunk Metadata 与 T09 术语映射验证",
        "",
        "## 结论",
        "",
        f"- 输入/输出 Chunk：{summary['input_chunks']} / {summary['output_chunks']}；映射追踪：{summary['trace_records']}。",
        f"- Schema 有效 Chunk/追踪：{summary['schema_valid_chunks']} / {summary['schema_valid_traces']}。",
        f"- Chunk ID 与正文等 T10.4 字段保持不变：{'是' if summary['immutable_fields_preserved'] else '否'}。",
        f"- 结构区段误加语义标签：{summary['excluded_section_tag_violations']} 条。",
        f"- 允许的多工序 Chunk：{summary['multi_process_chunks']} 条；多缺陷 Chunk：{summary['multi_defect_chunks']} 条。",
        f"- 全部验收检查：{'通过' if summary['acceptance_checks_passed'] else '未通过'}。",
        "",
        "## 标签覆盖",
        "",
        "| 来源 | Chunk | 排除结构区 | 有工序 | 有缺陷 | 有证据角色 | 多工序 | 多缺陷 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in summary["sources"]:
        lines.append(
            f"| {source['source_id']} | {source['chunk_count']} | "
            f"{source['excluded_structural_chunks']} | {source['chunks_with_process']} | "
            f"{source['chunks_with_defect']} | {source['chunks_with_evidence_role']} | "
            f"{source['multi_process_chunks']} | {source['multi_defect_chunks']} |"
        )
    lines.extend(["", "### 工序与缺陷", ""])
    lines.append(f"- 工序计数：`{summary['process_counts']}`")
    lines.append(f"- 缺陷计数：`{summary['defect_counts']}`")
    lines.append(f"- 证据角色计数：`{summary['evidence_role_counts']}`")
    lines.append(f"- 匹配类别计数：`{summary['match_category_counts']}`")
    lines.extend(["", "## 误匹配控制", ""])
    lines.append(
        f"- 禁用裸词命中：`{summary['disabled_bare_terms']}`；这些词在非结构 Chunk 中出现但被抑制的 Chunk 计数：`{summary['ambiguous_term_chunk_counts']}`。"
    )
    lines.append("- 英文词使用字母数字边界；短语按长度优先；封面、目录、广告、参考文献、作者和后置内容不加语义标签。")
    lines.append("- 候选原因命中只形成检索标签，并在追踪记录中保留关系 ID，不表示已确认根因。")
    lines.extend(["", "## 版本", ""])
    lines.append(f"- Chunk Schema：`{summary['chunk_schema_version']}`")
    lines.append(f"- Metadata Schema：`{summary['metadata_schema_version']}`")
    lines.append(f"- Mapping：`{summary['mapping_version']}`")
    lines.append(f"- Dictionary / Ontology：`{summary['dictionary_version']}` / `{summary['ontology_version']}`")
    lines.append(f"- Mapper：`{summary['mapper_version']}`")
    lines.append(f"- pip check：`{summary['environment']['pip_check_output']}`")
    lines.append(f"- 总耗时：`{summary['elapsed_seconds']}` 秒。")
    return "\n".join(lines) + "\n"


def run_metadata_pipeline(
    project_root: Path,
    source_ids: set[str] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    context = build_mapping_context(root)
    chunk_validator, trace_validator = load_metadata_validators(root)
    input_directory = root / context.config["input"]["chunks_directory"]
    input_paths = sorted(input_directory.glob("*.chunks.jsonl"))
    if source_ids:
        input_paths = [
            path
            for path in input_paths
            if path.name.removesuffix(".chunks.jsonl") in source_ids
        ]
        found = {path.name.removesuffix(".chunks.jsonl") for path in input_paths}
        missing = source_ids - found
        if missing:
            raise ValueError(f"Unknown or missing T10.4 sources: {sorted(missing)}")
    if not input_paths:
        raise FileNotFoundError(f"No T10.4 Chunk data found in {input_directory}")

    started = time.perf_counter()
    all_source_records: list[dict[str, Any]] = []
    all_enriched: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    ambiguous_counts: Counter[str] = Counter()

    for input_path in input_paths:
        source_id = input_path.name.removesuffix(".chunks.jsonl")
        source_records = _read_jsonl(input_path)
        enriched_records: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for source_record in source_records:
            enriched, trace = map_chunk(source_record, context)
            chunk_validator.validate(enriched)
            trace_validator.validate(trace)
            validate_metadata_invariants(source_record, enriched, trace)
            enriched_records.append(enriched)
            traces.append(trace)
            if not trace["excluded_from_semantic_tagging"]:
                searchable = normalize_for_match(
                    " > ".join(source_record["section_path"])
                    + "\n"
                    + source_record["text"]
                )
                for term in context.disabled_terms:
                    if contains_term(searchable, term):
                        ambiguous_counts[term] += 1

        output_path = (
            root
            / context.config["output"]["chunks_directory"]
            / f"{source_id}.chunks.v1.1.jsonl"
        )
        trace_path = (
            root
            / context.config["output"]["traces_directory"]
            / f"{source_id}.mapping-trace.jsonl"
        )
        _atomic_jsonl(output_path, enriched_records)
        _atomic_jsonl(trace_path, traces)
        source_summaries.append(
            _source_summary(
                source_id,
                enriched_records,
                traces,
                output_path,
                trace_path,
                root,
            )
        )
        all_source_records.extend(source_records)
        all_enriched.extend(enriched_records)
        all_traces.extend(traces)

    ids = [record["chunk_id"] for record in all_enriched]
    immutable_preserved = all(
        all(source.get(field) == enriched.get(field) for field in (
            "chunk_id",
            "source_id",
            "page_ids",
            "pdf_page_start",
            "pdf_page_end",
            "section_path",
            "text",
            "text_hash",
        ))
        for source, enriched in zip(all_source_records, all_enriched, strict=True)
    )
    excluded_violations = sum(
        trace["excluded_from_semantic_tagging"]
        and any(
            (
                trace["final_process_ids"],
                trace["final_defect_ids"],
                trace["final_evidence_roles"],
            )
        )
        for trace in all_traces
    )
    process_counts = Counter(
        value for record in all_enriched for value in record["metadata"]["process_ids"]
    )
    defect_counts = Counter(
        value for record in all_enriched for value in record["metadata"]["defect_ids"]
    )
    evidence_counts = Counter(
        value for record in all_enriched for value in record["metadata"]["evidence_roles"]
    )
    category_counts = Counter(
        match["term_category"] for trace in all_traces for match in trace["matches"]
    )
    acceptance = bool(
        len(all_source_records) == len(all_enriched) == len(all_traces)
        and len(ids) == len(set(ids))
        and immutable_preserved
        and not excluded_violations
        and all(record["metadata"]["dictionary_version"] == context.dictionary["dictionary_version"] for record in all_enriched)
        and all(record["metadata"]["ontology_version"] == context.ontology["ontology_version"] for record in all_enriched)
    )
    summary = {
        "schema_version": "1.0.0",
        "task": "T10.5",
        "chunk_schema_version": context.config["chunk_schema_version"],
        "metadata_schema_version": context.config["metadata_schema_version"],
        "mapping_version": context.config["mapping_version"],
        "mapper_version": context.config["mapper_version"],
        "dictionary_version": context.dictionary["dictionary_version"],
        "ontology_version": context.ontology["ontology_version"],
        "environment": collect_environment(),
        "total_sources": len(input_paths),
        "input_chunks": len(all_source_records),
        "output_chunks": len(all_enriched),
        "trace_records": len(all_traces),
        "schema_valid_chunks": len(all_enriched),
        "schema_valid_traces": len(all_traces),
        "unique_chunk_ids": len(set(ids)),
        "immutable_fields_preserved": immutable_preserved,
        "excluded_section_tag_violations": excluded_violations,
        "chunks_with_process": sum(bool(r["metadata"]["process_ids"]) for r in all_enriched),
        "chunks_with_defect": sum(bool(r["metadata"]["defect_ids"]) for r in all_enriched),
        "chunks_with_evidence_role": sum(bool(r["metadata"]["evidence_roles"]) for r in all_enriched),
        "multi_process_chunks": sum(len(r["metadata"]["process_ids"]) > 1 for r in all_enriched),
        "multi_defect_chunks": sum(len(r["metadata"]["defect_ids"]) > 1 for r in all_enriched),
        "process_counts": {key: process_counts.get(key, 0) for key in _PROCESS_ORDER},
        "defect_counts": {key: defect_counts.get(key, 0) for key in _DEFECT_ORDER},
        "evidence_role_counts": {key: evidence_counts.get(key, 0) for key in _EVIDENCE_ORDER},
        "match_category_counts": dict(sorted(category_counts.items())),
        "disabled_bare_terms": list(context.disabled_terms),
        "ambiguous_term_chunk_counts": {
            key: ambiguous_counts.get(key, 0) for key in context.disabled_terms
        },
        "sources": source_summaries,
        "acceptance_checks_passed": acceptance,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = root / context.config["output"]["summary_path"]
    _atomic_json(summary_path, summary)
    if write_report:
        report_path = root / context.config["output"]["report_path"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_metadata_report(summary), encoding="utf-8", newline="\n")
    return summary
