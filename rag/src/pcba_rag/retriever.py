from __future__ import annotations

import importlib.metadata
import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from jsonschema import Draft202012Validator, ValidationError
from qdrant_client import QdrantClient, models
from referencing import Registry, Resource

from .embedding_pipeline import (
    load_bge_m3_model,
    load_embedding_config,
    resolve_device,
    validate_local_model,
)
from .full_page_pipeline import _atomic_json
from .qdrant_pipeline import (
    create_qdrant_client,
    load_qdrant_config,
    wait_for_qdrant,
)


class RetrieverError(RuntimeError):
    """Base error raised by the framework-independent Retriever."""


class QueryTooLongError(RetrieverError):
    pass


class RetrieverUnavailableError(RetrieverError):
    pass


@dataclass(frozen=True)
class QueryVectors:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    token_count: int
    device: str


class QueryEncoder(Protocol):
    def encode_query(self, query: str) -> QueryVectors: ...


def load_retriever_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "rag/config/retriever.v0.3.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_retrieval_validator(project_root: Path) -> Draft202012Validator:
    metadata_schema = json.loads(
        (project_root / "rag/schemas/metadata.v1.1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    retrieval_schema = json.loads(
        (project_root / "rag/schemas/retrieval.v1.1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(metadata_schema)
    Draft202012Validator.check_schema(retrieval_schema)
    registry = Registry().with_resource(
        "metadata.v1.1.schema.json", Resource.from_contents(metadata_schema)
    )
    return Draft202012Validator(retrieval_schema, registry=registry)


def normalize_query(query: str, config: dict[str, Any]) -> str:
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    normalized = unicodedata.normalize(
        config["query"]["unicode_normalization"], query
    )
    if config["query"]["collapse_whitespace"]:
        normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()
    if not normalized:
        raise ValueError("Query is empty after normalization")
    return normalized


def _load_enabled_source_ids(project_root: Path, config: dict[str, Any]) -> set[str]:
    path = project_root / config["dependencies"]["sources_config"]
    source_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        source["source_id"]
        for source in source_config["sources"]
        if source.get("enabled", False)
    }


def validate_request(
    request: dict[str, Any],
    validator: Draft202012Validator,
    enabled_source_ids: set[str],
    config: dict[str, Any],
) -> None:
    try:
        validator.validate(request)
    except ValidationError as exc:
        raise ValueError(f"Invalid Retrieval V1.1 request: {exc.message}") from exc
    if request["top_k"] > int(config["retrieval"]["maximum_top_k"]):
        raise ValueError("top_k exceeds Retriever maximum_top_k")
    unknown_sources = set(request["filters"]["source_ids"]) - enabled_source_ids
    if unknown_sources:
        raise ValueError(f"Unknown source_ids: {sorted(unknown_sources)}")


FILTER_FIELD_MAP = {
    "source_ids": "source_id",
    "process_ids": "metadata.process_ids",
    "defect_ids": "metadata.defect_ids",
    "evidence_roles": "metadata.evidence_roles",
    "languages": "metadata.language",
    "document_types": "metadata.document_type",
}


def build_qdrant_filter(
    filters: dict[str, list[str]],
    config: dict[str, Any],
) -> models.Filter:
    conditions: list[models.FieldCondition] = []
    if config["retrieval"]["exclude_structural_content"]:
        conditions.append(
            models.FieldCondition(
                key="semantic_tag_excluded",
                match=models.MatchValue(value=False),
            )
        )
    for request_field, qdrant_field in FILTER_FIELD_MAP.items():
        values = filters[request_field]
        if values:
            conditions.append(
                models.FieldCondition(
                    key=qdrant_field,
                    match=models.MatchAny(any=values),
                )
            )
    return models.Filter(must=conditions)


class BgeM3QueryEncoder:
    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.project_root = project_root
        self.config = config
        self.embedding_config = load_embedding_config(project_root)
        self.model_directory = validate_local_model(project_root, self.embedding_config)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._use_fp16 = False

    def _load(self) -> None:
        if self._model is not None:
            return
        expected = str(self.embedding_config["model"]["implementation_version"])
        installed = importlib.metadata.version("FlagEmbedding")
        if installed != expected:
            raise RetrieverError(
                f"FlagEmbedding version mismatch: expected {expected}, got {installed}"
            )
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_directory, local_files_only=True
        )
        self._device, self._use_fp16 = resolve_device(self.embedding_config)
        self._model = load_bge_m3_model(
            self.model_directory,
            self.embedding_config,
            self._device,
            self._use_fp16,
        )

    def encode_query(self, query: str) -> QueryVectors:
        self._load()
        max_tokens = int(self.config["query"]["max_tokens"])
        token_count = len(self._tokenizer.encode(query, add_special_tokens=True))
        if token_count > max_tokens:
            raise QueryTooLongError(
                f"Query has {token_count} tokens; maximum is {max_tokens}"
            )
        output = self._model.encode_queries(
            [query],
            batch_size=int(self.config["query"]["batch_size"]),
            max_length=max_tokens,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(output["dense_vecs"][0], dtype=np.float32)
        expected_dimension = int(self.embedding_config["model"]["dense_dimension"])
        if dense.shape != (expected_dimension,) or not np.all(np.isfinite(dense)):
            raise RetrieverError(f"Unexpected query Dense vector shape {dense.shape}")
        raw_sparse = output["lexical_weights"][0]
        sparse_by_index = {
            int(index): float(np.float32(value))
            for index, value in raw_sparse.items()
            if float(value) > 0
        }
        indices = sorted(sparse_by_index)
        if not indices:
            raise RetrieverError("BGE-M3 returned an empty query Sparse vector")
        return QueryVectors(
            dense=dense.tolist(),
            sparse_indices=indices,
            sparse_values=[sparse_by_index[index] for index in indices],
            token_count=token_count,
            device=str(self._device),
        )


class Retriever:
    def __init__(
        self,
        project_root: Path,
        *,
        client: QdrantClient | Any | None = None,
        encoder: QueryEncoder | None = None,
        check_connection: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config = load_retriever_config(self.project_root)
        self.qdrant_config = load_qdrant_config(self.project_root)
        self.embedding_config = load_embedding_config(self.project_root)
        self.source_config = yaml.safe_load(
            (self.project_root / self.config["dependencies"]["sources_config"]).read_text(
                encoding="utf-8"
            )
        )
        self.validator = load_retrieval_validator(self.project_root)
        self.enabled_source_ids = _load_enabled_source_ids(
            self.project_root, self.config
        )
        self.client = client or create_qdrant_client(self.qdrant_config)
        self._owns_client = client is None
        self.encoder = encoder or BgeM3QueryEncoder(self.project_root, self.config)
        self._last_query: str | None = None
        self._last_vectors: QueryVectors | None = None
        if check_connection:
            try:
                wait_for_qdrant(self.client, self.qdrant_config)
                collection = self.qdrant_config["collection"]["name"]
                if not self.client.collection_exists(collection):
                    raise RetrieverUnavailableError(
                        f"Qdrant Collection is missing: {collection}"
                    )
            except RetrieverError:
                raise
            except Exception as exc:
                raise RetrieverUnavailableError(f"Qdrant is unavailable: {exc}") from exc

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _query_vectors(self, normalized_query: str) -> QueryVectors:
        if normalized_query != self._last_query or self._last_vectors is None:
            self._last_vectors = self.encoder.encode_query(normalized_query)
            self._last_query = normalized_query
        return self._last_vectors

    def retrieve_dense(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._retrieve(request, "dense")

    def retrieve_sparse(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._retrieve(request, "sparse")

    def retrieve_hybrid(self, request: dict[str, Any]) -> dict[str, Any]:
        from .fusion import retrieve_hybrid

        return retrieve_hybrid(self, request)

    def _retrieve(
        self,
        request: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        validate_request(
            request, self.validator, self.enabled_source_ids, self.config
        )
        normalized_query = normalize_query(request["query"], self.config)
        vectors = self._query_vectors(normalized_query)
        query_filter = build_qdrant_filter(request["filters"], self.config)
        dense_name = self.config["retrieval"]["dense_vector_name"]
        sparse_name = self.config["retrieval"]["sparse_vector_name"]
        if mode == "dense":
            query: list[float] | models.SparseVector = vectors.dense
            using = dense_name
        elif mode == "sparse":
            query = models.SparseVector(
                indices=vectors.sparse_indices, values=vectors.sparse_values
            )
            using = sparse_name
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        try:
            query_response = self.client.query_points(
                collection_name=self.qdrant_config["collection"]["name"],
                query=query,
                using=using,
                query_filter=query_filter,
                limit=request["top_k"],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise RetrieverUnavailableError(
                f"Qdrant {mode} retrieval failed: {exc}"
            ) from exc

        results = [
            self._result_from_point(point, rank, mode)
            for rank, point in enumerate(query_response.points, 1)
        ]
        response = {
            "schema_version": "1.1.0",
            "request_id": str(uuid.uuid4()),
            "query": request["query"],
            "normalized_query": normalized_query,
            "retrieval_mode": mode,
            "results": results,
            "trace": {
                "knowledge_base_version": self.source_config["knowledge_base_version"],
                "index_version": self.qdrant_config["index_version"],
                "collection_name": self.qdrant_config["collection"]["name"],
                "retriever_version": self.config["retriever_version"],
                "embedding_model": self.embedding_config["model"]["name"],
                "embedding_model_revision": self.embedding_config["model"][
                    "revision"
                ],
                "query_token_count": vectors.token_count,
                "applied_filters": request["filters"],
                "system_filters": {"semantic_tag_excluded": False},
                "retrieval_time_ms": round(
                    (time.perf_counter() - started) * 1000, 3
                ),
            },
        }
        try:
            self.validator.validate(response)
        except ValidationError as exc:
            raise RetrieverError(
                f"Generated Retrieval V1.1 response is invalid: {exc.message}"
            ) from exc
        return response

    def _result_from_point(
        self,
        point: Any,
        rank: int,
        mode: str,
    ) -> dict[str, Any]:
        payload = point.payload or {}
        expected_index = self.qdrant_config["index_version"]
        if payload.get("index_version") != expected_index:
            raise RetrieverError(
                f"Point {point.id} has unexpected index_version"
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise RetrieverError(f"Point {point.id} has no Metadata payload")
        score = float(point.score)
        return {
            "rank": rank,
            "chunk_id": payload["chunk_id"],
            "text": payload["text"],
            "citation": {
                "source_id": payload["source_id"],
                "source_title": metadata["source_title"],
                "pdf_page_start": payload["pdf_page_start"],
                "pdf_page_end": payload["pdf_page_end"],
                "section_path": payload["section_path"],
            },
            "metadata": metadata,
            "dense_score": score if mode == "dense" else None,
            "sparse_score": score if mode == "sparse" else None,
            "fusion_score": None,
            "rerank_score": None,
        }


def run_retriever_validation(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_retriever_config(root)
    case_results: list[dict[str, Any]] = []
    started = time.perf_counter()

    def result_matches_filters(
        result: dict[str, Any], filters: dict[str, list[str]]
    ) -> bool:
        metadata = result["metadata"]
        checks = [
            not filters["source_ids"]
            or result["citation"]["source_id"] in filters["source_ids"],
            not filters["process_ids"]
            or bool(set(metadata["process_ids"]) & set(filters["process_ids"])),
            not filters["defect_ids"]
            or bool(set(metadata["defect_ids"]) & set(filters["defect_ids"])),
            not filters["evidence_roles"]
            or bool(
                set(metadata["evidence_roles"]) & set(filters["evidence_roles"])
            ),
            not filters["languages"]
            or metadata["language"] in filters["languages"],
            not filters["document_types"]
            or metadata["document_type"] in filters["document_types"],
        ]
        return all(checks)

    with Retriever(root) as retriever:
        for case in config["validation"]["cases"]:
            request = {
                "schema_version": "1.1.0",
                "query": case["query"],
                "top_k": case["top_k"],
                "filters": case["filters"],
            }
            dense = retriever.retrieve_dense(request)
            sparse = retriever.retrieve_sparse(request)
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "normalized_query": dense["normalized_query"],
                    "query_token_count": dense["trace"]["query_token_count"],
                    "filters": case["filters"],
                    "dense": {
                        "result_count": len(dense["results"]),
                        "filter_compliance": all(
                            result_matches_filters(result, case["filters"])
                            for result in dense["results"]
                        ),
                        "top_chunk_ids": [
                            result["chunk_id"] for result in dense["results"]
                        ],
                        "scores": [
                            result["dense_score"] for result in dense["results"]
                        ],
                        "retrieval_time_ms": dense["trace"]["retrieval_time_ms"],
                    },
                    "sparse": {
                        "result_count": len(sparse["results"]),
                        "filter_compliance": all(
                            result_matches_filters(result, case["filters"])
                            for result in sparse["results"]
                        ),
                        "top_chunk_ids": [
                            result["chunk_id"] for result in sparse["results"]
                        ],
                        "scores": [
                            result["sparse_score"] for result in sparse["results"]
                        ],
                        "retrieval_time_ms": sparse["trace"]["retrieval_time_ms"],
                    },
                }
            )
    acceptance = bool(
        all(result["dense"]["result_count"] > 0 for result in case_results)
        and any(result["sparse"]["result_count"] > 0 for result in case_results)
        and all(
            result[mode]["filter_compliance"]
            for result in case_results
            for mode in ("dense", "sparse")
        )
    )
    summary = {
        "schema_version": "1.0.0",
        "task": "T10.8",
        "retrieval_schema_version": config["schema_version"],
        "retriever_version": config["retriever_version"],
        "collection_name": load_qdrant_config(root)["collection"]["name"],
        "query_model": load_embedding_config(root)["model"]["name"],
        "query_model_revision": load_embedding_config(root)["model"]["revision"],
        "validation_cases": case_results,
        "empty_sparse_cases": [
            result["case_id"]
            for result in case_results
            if result["sparse"]["result_count"] == 0
        ],
        "empty_results_are_valid": True,
        "total_runtime_seconds": round(time.perf_counter() - started, 3),
        "acceptance_checks_passed": acceptance,
    }
    _atomic_json(root / config["validation"]["summary_path"], summary)
    return summary
