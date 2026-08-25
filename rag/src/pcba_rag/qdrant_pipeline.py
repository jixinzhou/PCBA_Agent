from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from qdrant_client import QdrantClient, models

from .embedding_pipeline import _read_jsonl, sha256_text
from .full_page_pipeline import _atomic_json


@dataclass(frozen=True)
class QdrantIndexInput:
    chunk: dict[str, Any]
    embedding: dict[str, Any]


def load_qdrant_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "rag/config/qdrant.v0.3.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def point_id_for_chunk(chunk_id: str, namespace: str) -> str:
    if not chunk_id:
        raise ValueError("Chunk ID must not be empty")
    return str(uuid.uuid5(uuid.UUID(namespace), chunk_id))


def _validate_index_input(
    item: QdrantIndexInput,
    config: dict[str, Any],
) -> None:
    chunk = item.chunk
    embedding = item.embedding
    chunk_id = chunk["chunk_id"]
    if embedding["chunk_id"] != chunk_id:
        raise ValueError(f"Chunk/Embedding ID mismatch: {chunk_id}")
    if embedding["source_id"] != chunk["source_id"]:
        raise ValueError(f"Chunk/Embedding source mismatch: {chunk_id}")
    if embedding["text_hash"] != chunk["text_hash"]:
        raise ValueError(f"Chunk/Embedding text hash mismatch: {chunk_id}")
    if sha256_text(chunk["text"]) != chunk["text_hash"]:
        raise ValueError(f"Stored Chunk text hash is invalid: {chunk_id}")
    if embedding["model"]["revision"] != (
        "5617a9f61b028005a4858fdac845db406aefb181"
    ):
        raise ValueError(f"Unexpected BGE-M3 revision: {chunk_id}")

    dense = embedding["dense"]
    dimension = int(config["collection"]["dense_vector"]["size"])
    values = np.asarray(dense["values"], dtype=np.float32)
    if dense["dimension"] != dimension or values.shape != (dimension,):
        raise ValueError(f"Dense dimension mismatch: {chunk_id}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Dense vector contains non-finite values: {chunk_id}")

    sparse = embedding["sparse"]
    indices = sparse["indices"]
    sparse_values = np.asarray(sparse["values"], dtype=np.float32)
    if not indices or len(indices) != len(sparse_values) or len(indices) != sparse["nnz"]:
        raise ValueError(f"Invalid Sparse vector lengths: {chunk_id}")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError(f"Sparse indices are not sorted and unique: {chunk_id}")
    if not np.all(np.isfinite(sparse_values)) or np.any(sparse_values <= 0):
        raise ValueError(f"Sparse vector contains invalid values: {chunk_id}")


def load_index_inputs(
    project_root: Path,
    config: dict[str, Any],
) -> list[QdrantIndexInput]:
    chunks_directory = project_root / config["input"]["chunks_directory"]
    embeddings_directory = project_root / config["input"]["embeddings_directory"]
    chunk_paths = sorted(chunks_directory.glob("*.chunks.v1.1.jsonl"))
    if not chunk_paths:
        raise FileNotFoundError(f"No T10.5 enriched Chunks found in {chunks_directory}")

    items: list[QdrantIndexInput] = []
    seen_ids: set[str] = set()
    source_ids: set[str] = set()
    for chunk_path in chunk_paths:
        source_id = chunk_path.name.removesuffix(".chunks.v1.1.jsonl")
        source_ids.add(source_id)
        embedding_path = embeddings_directory / f"{source_id}.embeddings.jsonl"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing T10.6 Embedding data: {embedding_path}")
        chunks = _read_jsonl(chunk_path)
        embeddings = _read_jsonl(embedding_path)
        embeddings_by_id = {record["chunk_id"]: record for record in embeddings}
        if len(embeddings_by_id) != len(embeddings):
            raise ValueError(f"Duplicate Embedding Chunk IDs for {source_id}")
        chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        if chunk_ids != set(embeddings_by_id):
            raise ValueError(f"Chunk/Embedding ID set mismatch for {source_id}")
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate Chunk ID across sources: {chunk_id}")
            if chunk["source_id"] != source_id:
                raise ValueError(f"Chunk source differs from filename: {chunk_id}")
            item = QdrantIndexInput(chunk=chunk, embedding=embeddings_by_id[chunk_id])
            _validate_index_input(item, config)
            items.append(item)
            seen_ids.add(chunk_id)

    embedding_sources = {
        path.name.removesuffix(".embeddings.jsonl")
        for path in embeddings_directory.glob("*.embeddings.jsonl")
    }
    if embedding_sources != source_ids:
        raise ValueError("T10.5 Chunk sources and T10.6 Embedding sources differ")
    configured_count = config["validation"].get("require_exact_count")
    required_count = len(items) if configured_count is None else int(configured_count)
    if len(items) != required_count:
        raise ValueError(f"Expected {required_count} index inputs, got {len(items)}")
    return items


def build_payload(item: QdrantIndexInput, config: dict[str, Any]) -> dict[str, Any]:
    chunk = item.chunk
    embedding = item.embedding
    return {
        "index_version": config["index_version"],
        "chunk_schema_version": chunk["schema_version"],
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "page_ids": chunk["page_ids"],
        "pdf_page_start": chunk["pdf_page_start"],
        "pdf_page_end": chunk["pdf_page_end"],
        "section_path": chunk["section_path"],
        "text": chunk["text"],
        "text_hash": chunk["text_hash"],
        "metadata": chunk["metadata"],
        "semantic_tag_excluded": embedding["semantic_tag_excluded"],
        "embedding_input_hash": embedding["embedding_input_hash"],
        "embedding_input_token_count": embedding["embedding_input_token_count"],
        "embedding_model": embedding["model"]["name"],
        "embedding_model_revision": embedding["model"]["revision"],
        "embedding_config_version": embedding["model"][
            "embedding_config_version"
        ],
    }


def build_point(item: QdrantIndexInput, config: dict[str, Any]) -> models.PointStruct:
    dense_name = config["collection"]["dense_vector"]["name"]
    sparse_name = config["collection"]["sparse_vector"]["name"]
    sparse = item.embedding["sparse"]
    dense = np.asarray(item.embedding["dense"]["values"], dtype=np.float32)
    dense_norm = np.linalg.norm(dense)
    if not math.isfinite(float(dense_norm)) or dense_norm == 0:
        raise ValueError(f"Dense vector has invalid norm: {item.chunk['chunk_id']}")
    # Qdrant stores Cosine vectors L2-normalized. Normalize explicitly so the
    # expected storage fingerprint is identical to the server readback.
    stored_dense = np.asarray(dense / dense_norm, dtype=np.float32).tolist()
    return models.PointStruct(
        id=point_id_for_chunk(
            item.chunk["chunk_id"], config["point"]["uuid_namespace"]
        ),
        vector={
            dense_name: stored_dense,
            sparse_name: models.SparseVector(
                indices=sparse["indices"], values=sparse["values"]
            ),
        },
        payload=build_payload(item, config),
    )


def create_qdrant_client(config: dict[str, Any]) -> QdrantClient:
    return QdrantClient(
        url=config["qdrant"]["url"],
        timeout=int(config["qdrant"]["timeout_seconds"]),
        prefer_grpc=False,
        trust_env=False,
        check_compatibility=False,
    )


def wait_for_qdrant(client: QdrantClient, config: dict[str, Any]) -> str:
    attempts = int(config["qdrant"]["readiness_attempts"])
    interval = float(config["qdrant"]["readiness_interval_seconds"])
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            version = str(client.info().version)
            expected = str(config["qdrant"]["server_version"])
            if version != expected:
                raise RuntimeError(
                    f"Qdrant server version mismatch: expected {expected}, got {version}"
                )
            return version
        except RuntimeError:
            raise
        except Exception as exc:  # network error types vary by transport
            last_error = exc
            time.sleep(interval)
    raise ConnectionError(f"Qdrant did not become ready: {last_error}")


def _collection_schema_matches(
    client: QdrantClient,
    config: dict[str, Any],
) -> bool:
    info = client.get_collection(config["collection"]["name"])
    vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors
    dense_config = config["collection"]["dense_vector"]
    sparse_name = config["collection"]["sparse_vector"]["name"]
    if not isinstance(vectors, dict) or dense_config["name"] not in vectors:
        return False
    dense = vectors[dense_config["name"]]
    distance = getattr(dense.distance, "value", str(dense.distance))
    return bool(
        dense.size == int(dense_config["size"])
        and str(distance).lower() == str(dense_config["distance"]).lower()
        and isinstance(sparse_vectors, dict)
        and sparse_name in sparse_vectors
        and sparse_vectors[sparse_name].modifier is None
    )


def _create_collection(client: QdrantClient, config: dict[str, Any]) -> None:
    collection = config["collection"]
    dense = collection["dense_vector"]
    sparse = collection["sparse_vector"]
    client.create_collection(
        collection_name=collection["name"],
        vectors_config={
            dense["name"]: models.VectorParams(
                size=int(dense["size"]),
                distance=models.Distance(dense["distance"]),
                datatype=models.Datatype.FLOAT32,
            )
        },
        sparse_vectors_config={
            sparse["name"]: models.SparseVectorParams(
                index=models.SparseIndexParams(datatype=models.Datatype.FLOAT32),
                modifier=None,
            )
        },
        shard_number=int(collection["shard_number"]),
        replication_factor=int(collection["replication_factor"]),
        write_consistency_factor=int(collection["write_consistency_factor"]),
        on_disk_payload=bool(collection["on_disk_payload"]),
    )


def ensure_collection(
    client: QdrantClient,
    config: dict[str, Any],
    recreate: bool,
) -> tuple[bool, bool]:
    name = config["collection"]["name"]
    initially_existed = client.collection_exists(name)
    existed = initially_existed
    recreated = False
    if existed and recreate:
        client.delete_collection(name)
        existed = False
        recreated = True
    if existed and not _collection_schema_matches(client, config):
        raise ValueError(
            f"Existing Collection {name} has incompatible vector schema; "
            "use --recreate to replace this derived index"
        )
    if not existed:
        _create_collection(client, config)
    for field_name, raw_type in config["payload_indexes"].items():
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType(raw_type),
            wait=True,
        )
    return initially_existed, recreated


def _batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _scroll_all(
    client: QdrantClient,
    collection_name: str,
    *,
    with_payload: bool,
    with_vectors: bool,
    scroll_filter: models.Filter | None = None,
) -> list[models.Record]:
    records: list[models.Record] = []
    offset: int | str | uuid.UUID | None = None
    while True:
        page, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        records.extend(page)
        if offset is None:
            return records


def upsert_and_sync_points(
    client: QdrantClient,
    items: list[QdrantIndexInput],
    config: dict[str, Any],
) -> int:
    name = config["collection"]["name"]
    points = [build_point(item, config) for item in items]
    batch_size = int(config["point"]["batch_size"])
    for batch in _batches(points, batch_size):
        client.upsert(collection_name=name, points=batch, wait=True)
    expected_ids = {str(point.id) for point in points}
    actual_ids = {
        str(record.id)
        for record in _scroll_all(
            client, name, with_payload=False, with_vectors=False
        )
    }
    stale_ids = sorted(actual_ids - expected_ids)
    if stale_ids:
        client.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=stale_ids),
            wait=True,
        )
    return len(stale_ids)


def _vector_parts(vector: Any, dense_name: str, sparse_name: str) -> tuple[list[float], list[int], list[float]]:
    if not isinstance(vector, dict):
        raise ValueError("Qdrant Point does not contain named vectors")
    dense = list(vector[dense_name])
    sparse = vector[sparse_name]
    if isinstance(sparse, dict):
        indices = list(sparse["indices"])
        values = list(sparse["values"])
    else:
        indices = list(sparse.indices)
        values = list(sparse.values)
    return dense, indices, values


def _content_fingerprint(
    point_id: str,
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    payload: dict[str, Any],
) -> bytes:
    digest = hashlib.sha256()
    digest.update(point_id.encode("ascii"))
    digest.update(np.asarray(dense, dtype="<f4").tobytes())
    digest.update(np.asarray(sparse_indices, dtype="<u8").tobytes())
    digest.update(np.asarray(sparse_values, dtype="<f4").tobytes())
    digest.update(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return digest.digest()


def index_fingerprint(points: Iterable[Any], config: dict[str, Any]) -> str:
    dense_name = config["collection"]["dense_vector"]["name"]
    sparse_name = config["collection"]["sparse_vector"]["name"]
    pieces: list[tuple[str, bytes]] = []
    for point in points:
        dense, indices, values = _vector_parts(
            point.vector, dense_name, sparse_name
        )
        point_id = str(point.id)
        pieces.append(
            (
                point_id,
                _content_fingerprint(
                    point_id, dense, indices, values, point.payload or {}
                ),
            )
        )
    digest = hashlib.sha256()
    for point_id, content in sorted(pieces):
        digest.update(point_id.encode("ascii"))
        digest.update(content)
    return digest.hexdigest()


def validate_collection(
    client: QdrantClient,
    items: list[QdrantIndexInput],
    config: dict[str, Any],
) -> dict[str, Any]:
    name = config["collection"]["name"]
    if not _collection_schema_matches(client, config):
        raise ValueError("Collection vector schema differs from T10.7 configuration")
    expected_points = [build_point(item, config) for item in items]
    actual_points = _scroll_all(
        client, name, with_payload=True, with_vectors=True
    )
    exact_count = client.count(collection_name=name, exact=True).count
    expected_by_id = {str(point.id): point for point in expected_points}
    actual_by_id = {str(point.id): point for point in actual_points}
    if exact_count != len(items) or set(expected_by_id) != set(actual_by_id):
        raise ValueError("Qdrant Point count or ID set differs from index input")

    sample_size = min(int(config["validation"]["sample_size"]), len(items))
    positions = (
        [0]
        if sample_size == 1
        else [
            round(index * (len(items) - 1) / (sample_size - 1))
            for index in range(sample_size)
        ]
    )
    tolerance = float(config["validation"]["vector_absolute_tolerance"])
    dense_name = config["collection"]["dense_vector"]["name"]
    sparse_name = config["collection"]["sparse_vector"]["name"]
    for position in positions:
        expected = expected_points[position]
        actual = actual_by_id[str(expected.id)]
        expected_dense, expected_indices, expected_sparse = _vector_parts(
            expected.vector, dense_name, sparse_name
        )
        actual_dense, actual_indices, actual_sparse = _vector_parts(
            actual.vector, dense_name, sparse_name
        )
        if not np.allclose(
            expected_dense, actual_dense, rtol=0.0, atol=tolerance
        ):
            raise ValueError(f"Dense readback mismatch for Point {expected.id}")
        if expected_indices != actual_indices or not np.allclose(
            expected_sparse, actual_sparse, rtol=0.0, atol=tolerance
        ):
            raise ValueError(f"Sparse readback mismatch for Point {expected.id}")
        if expected.payload != actual.payload:
            raise ValueError(f"Payload readback mismatch for Point {expected.id}")

    first = expected_points[0]
    dense, indices, sparse_values = _vector_parts(
        first.vector, dense_name, sparse_name
    )
    dense_result = client.query_points(
        collection_name=name,
        query=dense,
        using=dense_name,
        limit=1,
        with_payload=False,
    )
    sparse_result = client.query_points(
        collection_name=name,
        query=models.SparseVector(indices=indices, values=sparse_values),
        using=sparse_name,
        limit=1,
        with_payload=False,
    )
    dense_self_match = bool(
        dense_result.points and str(dense_result.points[0].id) == str(first.id)
    )
    sparse_self_match = bool(
        sparse_result.points and str(sparse_result.points[0].id) == str(first.id)
    )
    if not dense_self_match or not sparse_self_match:
        raise ValueError("Dense or Sparse self-query did not return the source Point")

    info = client.get_collection(name)
    payload_fields = set(info.payload_schema)
    expected_payload_fields = set(config["payload_indexes"])
    if not expected_payload_fields.issubset(payload_fields):
        raise ValueError("One or more configured Payload indexes are missing")

    def payload_value(payload: dict[str, Any], field_name: str) -> Any:
        value: Any = payload
        for part in field_name.split("."):
            value = value[part]
        return value

    filter_checks: dict[str, dict[str, Any]] = {}
    expected_payloads = [point.payload or {} for point in expected_points]
    for field_name in config["payload_indexes"]:
        candidates = [payload_value(payload, field_name) for payload in expected_payloads]
        nonempty = [value for value in candidates if value not in (None, [], "")]
        if not nonempty:
            raise ValueError(f"No testable value for Payload index {field_name}")
        sample_value = nonempty[0][0] if isinstance(nonempty[0], list) else nonempty[0]
        expected_count = sum(
            sample_value in value if isinstance(value, list) else value == sample_value
            for value in candidates
        )
        actual_count = client.count(
            collection_name=name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=field_name,
                        match=models.MatchValue(value=sample_value),
                    )
                ]
            ),
            exact=True,
        ).count
        if actual_count != expected_count:
            raise ValueError(f"Payload filter count mismatch for {field_name}")
        filter_checks[field_name] = {
            "sample_value": sample_value,
            "expected_count": int(expected_count),
            "actual_count": int(actual_count),
            "passed": True,
        }

    semantic_false = sum(
        not item.embedding["semantic_tag_excluded"] for item in items
    )

    expected_fingerprint = index_fingerprint(expected_points, config)
    actual_fingerprint = index_fingerprint(actual_points, config)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError("Qdrant content fingerprint differs from source records")
    return {
        "exact_point_count": int(exact_count),
        "unique_point_ids": len(actual_by_id),
        "sample_readback_count": sample_size,
        "dense_self_query_passed": dense_self_match,
        "sparse_self_query_passed": sparse_self_match,
        "payload_index_fields": sorted(payload_fields),
        "payload_filter_checks": filter_checks,
        "semantic_tag_included_points": int(semantic_false),
        "semantic_tag_excluded_points": len(items) - int(semantic_false),
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual_fingerprint,
        "fingerprint_matches": True,
    }


def run_qdrant_pipeline(
    project_root: Path,
    *,
    recreate: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = project_root.resolve()
    config = load_qdrant_config(root)
    installed_client = importlib.metadata.version("qdrant-client")
    if installed_client != str(config["qdrant"]["client_version"]):
        raise RuntimeError(
            "qdrant-client version mismatch: "
            f"expected {config['qdrant']['client_version']}, got {installed_client}"
        )
    items = load_index_inputs(root, config)
    client = create_qdrant_client(config)
    try:
        server_version = wait_for_qdrant(client, config)
        existed, recreated = ensure_collection(client, config, recreate)
        stale_removed = upsert_and_sync_points(client, items, config)
        validation = validate_collection(client, items, config)
    finally:
        client.close()

    summary = {
        "schema_version": "1.0.0",
        "task": "T10.7",
        "index_version": config["index_version"],
        "pipeline_version": config["pipeline_version"],
        "collection_name": config["collection"]["name"],
        "qdrant_url": config["qdrant"]["url"],
        "qdrant_server_version": server_version,
        "qdrant_client_version": installed_client,
        "collection_existed": existed,
        "collection_recreated": recreated,
        "stale_points_removed": stale_removed,
        "input_records": len(items),
        "dense_vector": config["collection"]["dense_vector"],
        "sparse_vector": config["collection"]["sparse_vector"],
        "point_id_strategy": "UUIDv5(fixed_namespace, chunk_id)",
        "uuid_namespace": config["point"]["uuid_namespace"],
        "validation": validation,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "acceptance_checks_passed": bool(
            validation["exact_point_count"] == len(items)
            and validation["unique_point_ids"] == len(items)
            and validation["fingerprint_matches"]
            and validation["dense_self_query_passed"]
            and validation["sparse_self_query_passed"]
        ),
    }
    _atomic_json(root / config["output"]["summary_path"], summary)
    return summary
