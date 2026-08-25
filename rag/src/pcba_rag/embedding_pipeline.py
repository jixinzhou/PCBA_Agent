from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from jsonschema import Draft202012Validator

from .full_page_pipeline import _atomic_json
from .sample_pipeline import collect_environment


class EmbeddingModel(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> dict[str, Any]: ...


def load_embedding_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "rag/config/embedding.v0.1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_embedding_validator(project_root: Path) -> Draft202012Validator:
    schema = json.loads(
        (project_root / "rag/schemas/embedding_record.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def build_embedding_text(chunk: dict[str, Any], config: dict[str, Any]) -> str:
    section = config["input"]["section_separator"].join(chunk["section_path"])
    values = [value for value in (section.strip(), chunk["text"].strip()) if value]
    return config["input"]["section_body_separator"].join(values)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _load_source_inputs(
    project_root: Path,
    config: dict[str, Any],
    source_ids: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    chunks_directory = project_root / config["input"]["chunks_directory"]
    traces_directory = project_root / config["input"]["traces_directory"]
    paths = sorted(chunks_directory.glob("*.chunks.v1.1.jsonl"))
    if source_ids:
        paths = [
            path
            for path in paths
            if path.name.removesuffix(".chunks.v1.1.jsonl") in source_ids
        ]
        found = {
            path.name.removesuffix(".chunks.v1.1.jsonl") for path in paths
        }
        missing = source_ids - found
        if missing:
            raise ValueError(f"Unknown or missing T10.5 sources: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"No T10.5 Chunk data found in {chunks_directory}")

    chunks: list[dict[str, Any]] = []
    traces: dict[str, dict[str, Any]] = {}
    source_order: list[str] = []
    for path in paths:
        source_id = path.name.removesuffix(".chunks.v1.1.jsonl")
        trace_path = traces_directory / f"{source_id}.mapping-trace.jsonl"
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing T10.5 mapping trace: {trace_path}")
        source_chunks = _read_jsonl(path)
        source_traces = _read_jsonl(trace_path)
        if len(source_chunks) != len(source_traces):
            raise ValueError(f"Chunk/trace count mismatch for {source_id}")
        for trace in source_traces:
            chunk_id = trace["chunk_id"]
            if chunk_id in traces:
                raise ValueError(f"Duplicate trace for {chunk_id}")
            traces[chunk_id] = trace
        for chunk in source_chunks:
            if chunk["source_id"] != source_id:
                raise ValueError(f"Source mismatch in {path}: {chunk['chunk_id']}")
            if chunk["chunk_id"] not in traces:
                raise ValueError(f"Missing trace for {chunk['chunk_id']}")
        chunks.extend(source_chunks)
        source_order.append(source_id)
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate Chunk IDs in T10.5 input")
    return chunks, traces, source_order


def _required_model_files() -> set[str]:
    return {
        "config.json",
        "pytorch_model.bin",
        "colbert_linear.pt",
        "sparse_linear.pt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }


def validate_local_model(project_root: Path, config: dict[str, Any]) -> Path:
    model_directory = project_root / config["model"]["local_directory"]
    present = (
        {path.name for path in model_directory.iterdir() if path.is_file()}
        if model_directory.exists()
        else set()
    )
    missing = _required_model_files() - present
    if missing:
        raise FileNotFoundError(
            f"Incomplete local BGE-M3 snapshot at {model_directory}; missing {sorted(missing)}"
        )
    return model_directory


def resolve_device(config: dict[str, Any]) -> tuple[str, bool]:
    import torch

    preferred = config["model"]["preferred_device"]
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return preferred, bool(config["model"]["use_fp16_on_cuda"])
    if not config["model"].get("allow_cpu_fallback", False):
        raise RuntimeError(f"Preferred device {preferred} is unavailable")
    return "cpu", False


def load_bge_m3_model(
    model_directory: Path,
    config: dict[str, Any],
    device: str,
    use_fp16: bool,
) -> EmbeddingModel:
    installed = importlib.metadata.version("FlagEmbedding")
    expected = str(config["model"]["implementation_version"])
    if installed != expected:
        raise RuntimeError(
            f"FlagEmbedding version mismatch: expected {expected}, got {installed}"
        )
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(
        str(model_directory),
        normalize_embeddings=bool(config["model"]["normalize_embeddings"]),
        use_fp16=use_fp16,
        devices=device,
        batch_size=int(config["model"]["batch_size"]),
        passage_max_length=int(config["model"]["max_length"]),
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )


def _encode(
    model: EmbeddingModel,
    texts: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    output = model.encode(
        texts,
        batch_size=int(config["model"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = np.asarray(output["dense_vecs"], dtype=np.float32)
    sparse = output["lexical_weights"]
    if dense.ndim != 2 or dense.shape[0] != len(texts):
        raise ValueError(f"Unexpected Dense shape {dense.shape} for {len(texts)} texts")
    if len(sparse) != len(texts):
        raise ValueError("Sparse output count differs from input count")
    return dense, sparse


def _sparse_payload(weights: dict[str, float]) -> dict[str, Any]:
    values_by_index: dict[int, float] = {}
    for raw_index, raw_value in weights.items():
        index = int(raw_index)
        value = float(np.float32(raw_value))
        if value > 0:
            values_by_index[index] = value
    indices = sorted(values_by_index)
    values = [values_by_index[index] for index in indices]
    return {"indices": indices, "values": values, "nnz": len(indices)}


def build_embedding_records(
    chunks: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    texts: list[str],
    token_counts: list[int],
    dense_vectors: np.ndarray,
    sparse_vectors: list[dict[str, float]],
    config: dict[str, Any],
    device: str,
    use_fp16: bool,
) -> list[dict[str, Any]]:
    if not (
        len(chunks)
        == len(texts)
        == len(token_counts)
        == len(dense_vectors)
        == len(sparse_vectors)
    ):
        raise ValueError("Embedding inputs and outputs have inconsistent counts")
    model_metadata = {
        "name": config["model"]["name"],
        "revision": config["model"]["revision"],
        "embedding_config_version": config["embedding_config_version"],
        "implementation": config["model"]["implementation"],
        "implementation_version": str(config["model"]["implementation_version"]),
        "device": device,
        "inference_dtype": "float16" if use_fp16 else "float32",
        "storage_dtype": config["output"]["storage_dtype"],
        "max_length": int(config["model"]["max_length"]),
        "normalized": bool(config["model"]["normalize_embeddings"]),
    }
    records: list[dict[str, Any]] = []
    for chunk, text, token_count, dense, sparse in zip(
        chunks,
        texts,
        token_counts,
        dense_vectors,
        sparse_vectors,
        strict=True,
    ):
        records.append(
            {
                "schema_version": "1.0.0",
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "text_hash": chunk["text_hash"],
                "embedding_input_hash": sha256_text(text),
                "embedding_input_token_count": token_count,
                "semantic_tag_excluded": bool(
                    traces[chunk["chunk_id"]]["excluded_from_semantic_tagging"]
                ),
                "model": dict(model_metadata),
                "dense": {
                    "dimension": int(dense.shape[0]),
                    "values": np.asarray(dense, dtype=np.float32).tolist(),
                },
                "sparse": _sparse_payload(sparse),
            }
        )
    return records


def validate_embedding_invariants(
    chunk: dict[str, Any],
    trace: dict[str, Any],
    record: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if record["chunk_id"] != chunk["chunk_id"]:
        raise ValueError("Embedding record Chunk ID mismatch")
    if record["source_id"] != chunk["source_id"]:
        raise ValueError("Embedding record source mismatch")
    if record["text_hash"] != chunk["text_hash"]:
        raise ValueError("Embedding record text hash mismatch")
    if sha256_text(chunk["text"]) != chunk["text_hash"]:
        raise ValueError(f"Stored Chunk text hash is invalid: {chunk['chunk_id']}")
    expected_text = build_embedding_text(chunk, config)
    if record["embedding_input_hash"] != sha256_text(expected_text):
        raise ValueError("Embedding input hash mismatch")
    if record["embedding_input_token_count"] > config["model"]["max_length"]:
        raise ValueError("Embedding input was truncated")
    if record["semantic_tag_excluded"] != bool(
        trace["excluded_from_semantic_tagging"]
    ):
        raise ValueError("Structural content flag differs from T10.5 trace")

    dense = np.asarray(record["dense"]["values"], dtype=np.float32)
    expected_dimension = int(config["model"]["dense_dimension"])
    if dense.shape != (expected_dimension,):
        raise ValueError(f"Dense dimension mismatch: {dense.shape}")
    if not np.all(np.isfinite(dense)):
        raise ValueError("Dense vector contains NaN or Infinity")
    norm = float(np.linalg.norm(dense))
    tolerance = float(config["validation"]["dense_norm_tolerance"])
    if bool(config["model"]["normalize_embeddings"]) and abs(norm - 1.0) > tolerance:
        raise ValueError(f"Dense vector norm {norm:.6f} is outside tolerance")

    sparse = record["sparse"]
    indices = sparse["indices"]
    values = np.asarray(sparse["values"], dtype=np.float32)
    if len(indices) != len(values) or len(indices) != sparse["nnz"]:
        raise ValueError("Sparse indices, values and nnz differ")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("Sparse indices are not sorted and unique")
    if bool(config["validation"]["require_nonempty_sparse"]) and not indices:
        raise ValueError("Sparse vector is empty")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Sparse values must be finite and positive")


def _reproducibility_check(
    model: EmbeddingModel,
    texts: list[str],
    dense_vectors: np.ndarray,
    sparse_vectors: list[dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    sample_size = min(
        int(config["validation"]["reproducibility_sample_size"]), len(texts)
    )
    if sample_size == 0:
        return {"sample_size": 0, "dense_allclose": True, "sparse_allclose": True}
    if sample_size == 1:
        positions = [0]
    else:
        positions = [
            round(index * (len(texts) - 1) / (sample_size - 1))
            for index in range(sample_size)
        ]
    repeated_dense, repeated_sparse = _encode(
        model, [texts[index] for index in positions], config
    )
    expected_dense = dense_vectors[positions]
    atol = float(config["validation"]["reproducibility_atol"])
    dense_allclose = bool(
        np.allclose(expected_dense, repeated_dense, rtol=0.0, atol=atol)
    )
    sparse_allclose = True
    for output_index, source_index in enumerate(positions):
        first = _sparse_payload(sparse_vectors[source_index])
        second = _sparse_payload(repeated_sparse[output_index])
        if first["indices"] != second["indices"] or not np.allclose(
            first["values"], second["values"], rtol=0.0, atol=atol
        ):
            sparse_allclose = False
            break
    return {
        "sample_size": sample_size,
        "positions": positions,
        "dense_allclose": dense_allclose,
        "sparse_allclose": bool(sparse_allclose),
        "absolute_tolerance": atol,
    }


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    position = max(0, math.ceil(fraction * len(ordered)) - 1)
    return int(ordered[position])


def render_embedding_report(summary: dict[str, Any]) -> str:
    token = summary["token_statistics"]
    sparse = summary["sparse_nnz_statistics"]
    lines = [
        "# T10.6 BGE-M3 Embedding 验证",
        "",
        "## 结论",
        "",
        f"- 输入 Chunk / Embedding：{summary['input_chunks']} / {summary['embedding_records']}。",
        f"- Schema 有效：{summary['schema_valid_records']}；唯一 Chunk ID：{summary['unique_chunk_ids']}。",
        f"- Dense：{summary['dense_dimension']} 维；Sparse 非空：{summary['nonempty_sparse_records']} 条。",
        f"- 最大输入：{token['maximum']} tokens；截断：{summary['truncated_inputs']} 条。",
        f"- 结构性内容：{summary['semantic_tag_excluded_records']} 条，全部保留标记。",
        f"- 重复编码检查：Dense {'通过' if summary['reproducibility']['dense_allclose'] else '未通过'}，Sparse {'通过' if summary['reproducibility']['sparse_allclose'] else '未通过'}。",
        f"- 全部验收检查：{'通过' if summary['acceptance_checks_passed'] else '未通过'}。",
        "",
        "## 固定配置",
        "",
        f"- 模型：`{summary['model']['name']}`",
        f"- revision：`{summary['model']['revision']}`",
        f"- 实现：`{summary['model']['implementation']} {summary['model']['implementation_version']}`",
        f"- 设备：`{summary['model']['device']}`；推理精度：`{summary['model']['inference_dtype']}`；保存精度：`{summary['model']['storage_dtype']}`。",
        f"- max_length：`{summary['model']['max_length']}`；batch_size：`{summary['model']['batch_size']}`；ColBERT：关闭。",
        f"- 输入格式：`section_path + 两个换行 + text`；Metadata 标签不注入模型文本。",
        "",
        "## 统计",
        "",
        f"- Token：最小 {token['minimum']}，P50 {token['p50']}，P95 {token['p95']}，最大 {token['maximum']}。",
        f"- Sparse NNZ：最小 {sparse['minimum']}，P50 {sparse['p50']}，P95 {sparse['p95']}，最大 {sparse['maximum']}。",
        f"- 每来源记录：`{summary['source_counts']}`",
        f"- GPU 峰值已分配显存：`{summary['runtime'].get('peak_gpu_allocated_mb')}` MB。",
        f"- Embedding 耗时：`{summary['runtime']['embedding_seconds']}` 秒；总耗时：`{summary['runtime']['total_seconds']}` 秒。",
        "",
        "## 完整性检查",
        "",
        "- 每条记录保存原始 text_hash、Embedding 输入哈希、模型 revision、配置版本、设备和精度。",
        "- Dense 数值均有限且归一化；Sparse 索引排序且唯一，权重均为有限正数。",
        f"- pip check：`{summary['environment']['pip_check_output']}`",
    ]
    return "\n".join(lines) + "\n"


def run_embedding_pipeline(
    project_root: Path,
    source_ids: set[str] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_embedding_config(root)
    validator = load_embedding_validator(root)
    model_directory = validate_local_model(root, config)
    chunks, traces, source_order = _load_source_inputs(root, config, source_ids)
    texts = [build_embedding_text(chunk, config) for chunk in chunks]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
    token_counts = [
        len(tokenizer.encode(text, add_special_tokens=True)) for text in texts
    ]
    max_length = int(config["model"]["max_length"])
    truncated_inputs = sum(value > max_length for value in token_counts)
    if config["validation"]["require_zero_truncation"] and truncated_inputs:
        raise ValueError(
            f"{truncated_inputs} Embedding inputs exceed max_length={max_length}"
        )

    started = time.perf_counter()
    device, use_fp16 = resolve_device(config)
    import torch

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_bge_m3_model(model_directory, config, device, use_fp16)
    embedding_started = time.perf_counter()
    try:
        dense_vectors, sparse_vectors = _encode(model, texts, config)
    except RuntimeError as exc:
        is_cuda_oom = device.startswith("cuda") and "out of memory" in str(exc).lower()
        if not is_cuda_oom or not config["model"].get("allow_cpu_fallback", False):
            raise
        del model
        gc.collect()
        torch.cuda.empty_cache()
        device, use_fp16 = "cpu", False
        model = load_bge_m3_model(model_directory, config, device, use_fp16)
        dense_vectors, sparse_vectors = _encode(model, texts, config)
    embedding_seconds = round(time.perf_counter() - embedding_started, 3)

    expected_dimension = int(config["model"]["dense_dimension"])
    if dense_vectors.shape != (len(chunks), expected_dimension):
        raise ValueError(f"Unexpected Dense matrix shape {dense_vectors.shape}")
    reproducibility = _reproducibility_check(
        model, texts, dense_vectors, sparse_vectors, config
    )
    records = build_embedding_records(
        chunks,
        traces,
        texts,
        token_counts,
        dense_vectors,
        sparse_vectors,
        config,
        device,
        use_fp16,
    )

    schema_valid = 0
    for chunk, record in zip(chunks, records, strict=True):
        validator.validate(record)
        validate_embedding_invariants(
            chunk, traces[chunk["chunk_id"]], record, config
        )
        schema_valid += 1

    records_by_source: dict[str, list[dict[str, Any]]] = {
        source_id: [] for source_id in source_order
    }
    for record in records:
        records_by_source[record["source_id"]].append(record)
    output_directory = root / config["output"]["embeddings_directory"]
    output_paths: dict[str, str] = {}
    for source_id in source_order:
        path = output_directory / f"{source_id}.embeddings.jsonl"
        _atomic_jsonl(path, records_by_source[source_id])
        output_paths[source_id] = str(path.relative_to(root))

    dense_norms = [
        float(np.linalg.norm(np.asarray(record["dense"]["values"], dtype=np.float32)))
        for record in records
    ]
    sparse_counts = [record["sparse"]["nnz"] for record in records]
    ids = [record["chunk_id"] for record in records]
    source_counts = Counter(record["source_id"] for record in records)
    peak_gpu_allocated_mb = None
    peak_gpu_reserved_mb = None
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if device.startswith("cuda"):
            peak_gpu_allocated_mb = round(
                torch.cuda.max_memory_allocated() / 1024**2, 2
            )
            peak_gpu_reserved_mb = round(
                torch.cuda.max_memory_reserved() / 1024**2, 2
            )

    environment = collect_environment()
    environment["packages"]["FlagEmbedding"] = importlib.metadata.version(
        "FlagEmbedding"
    )
    acceptance = bool(
        len(chunks) == len(records)
        and schema_valid == len(records)
        and len(ids) == len(set(ids))
        and not truncated_inputs
        and all(record["sparse"]["nnz"] > 0 for record in records)
        and all(math.isfinite(value) for value in dense_norms)
        and reproducibility["dense_allclose"]
        and reproducibility["sparse_allclose"]
    )
    summary = {
        "schema_version": "1.0.0",
        "task": "T10.6",
        "embedding_record_schema_version": "1.0.0",
        "embedding_config_version": config["embedding_config_version"],
        "pipeline_version": config["pipeline_version"],
        "model": {
            "name": config["model"]["name"],
            "revision": config["model"]["revision"],
            "local_directory": config["model"]["local_directory"],
            "implementation": config["model"]["implementation"],
            "implementation_version": str(
                config["model"]["implementation_version"]
            ),
            "device": device,
            "gpu_name": gpu_name,
            "inference_dtype": "float16" if use_fp16 else "float32",
            "storage_dtype": config["output"]["storage_dtype"],
            "max_length": max_length,
            "batch_size": int(config["model"]["batch_size"]),
            "normalize_embeddings": bool(
                config["model"]["normalize_embeddings"]
            ),
            "return_dense": True,
            "return_sparse": True,
            "return_colbert_vecs": False,
        },
        "environment": environment,
        "input_sources": len(source_order),
        "input_chunks": len(chunks),
        "embedding_records": len(records),
        "schema_valid_records": schema_valid,
        "unique_chunk_ids": len(set(ids)),
        "dense_dimension": expected_dimension,
        "dense_norm_statistics": {
            "minimum": round(min(dense_norms), 6),
            "maximum": round(max(dense_norms), 6),
        },
        "nonempty_sparse_records": sum(value > 0 for value in sparse_counts),
        "token_statistics": {
            "minimum": min(token_counts),
            "p50": _percentile(token_counts, 0.50),
            "p95": _percentile(token_counts, 0.95),
            "maximum": max(token_counts),
        },
        "sparse_nnz_statistics": {
            "minimum": min(sparse_counts),
            "p50": _percentile(sparse_counts, 0.50),
            "p95": _percentile(sparse_counts, 0.95),
            "maximum": max(sparse_counts),
        },
        "truncated_inputs": truncated_inputs,
        "semantic_tag_excluded_records": sum(
            record["semantic_tag_excluded"] for record in records
        ),
        "source_counts": {
            source_id: source_counts[source_id] for source_id in source_order
        },
        "output_paths": output_paths,
        "reproducibility": reproducibility,
        "runtime": {
            "embedding_seconds": embedding_seconds,
            "total_seconds": round(time.perf_counter() - started, 3),
            "peak_gpu_allocated_mb": peak_gpu_allocated_mb,
            "peak_gpu_reserved_mb": peak_gpu_reserved_mb,
        },
        "acceptance_checks_passed": acceptance,
    }
    summary_path = root / config["output"]["summary_path"]
    _atomic_json(summary_path, summary)
    if write_report:
        report_path = root / config["output"]["report_path"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_embedding_report(summary), encoding="utf-8", newline="\n"
        )
    return summary
