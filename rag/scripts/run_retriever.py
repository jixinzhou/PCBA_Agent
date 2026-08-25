from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run framework-independent Dense/Sparse/Hybrid retrieval"
    )
    parser.add_argument("query")
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "both", "hybrid", "all"],
        default="hybrid",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--process-id", action="append", default=[])
    parser.add_argument("--defect-id", action="append", default=[])
    parser.add_argument("--evidence-role", action="append", default=[])
    parser.add_argument("--language", action="append", default=[])
    parser.add_argument("--document-type", action="append", default=[])
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "rag/src"))
    from pcba_rag.fusion import load_fusion_config
    from pcba_rag.retriever import Retriever, load_retriever_config

    retriever_config = load_retriever_config(root)
    fusion_config = load_fusion_config(root)
    channel_top_k = args.top_k or int(
        retriever_config["retrieval"]["default_top_k"]
    )
    final_top_k = args.top_k or int(
        fusion_config["ranking"]["default_final_top_k"]
    )
    filters = {
        "source_ids": args.source_id,
        "process_ids": args.process_id,
        "defect_ids": args.defect_id,
        "evidence_roles": args.evidence_role,
        "languages": args.language,
        "document_types": args.document_type,
    }
    channel_request = {
        "schema_version": "1.1.0",
        "query": args.query,
        "top_k": channel_top_k,
        "filters": filters,
    }
    hybrid_request = {
        "schema_version": "1.2.0",
        "query": args.query,
        "top_k": final_top_k,
        "filters": filters,
    }
    with Retriever(root) as retriever:
        output: dict[str, object] = {}
        if args.mode in ("dense", "both", "all"):
            output["dense"] = retriever.retrieve_dense(channel_request)
        if args.mode in ("sparse", "both", "all"):
            output["sparse"] = retriever.retrieve_sparse(channel_request)
        if args.mode in ("hybrid", "all"):
            output["hybrid"] = retriever.retrieve_hybrid(hybrid_request)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
