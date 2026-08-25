from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


class KGAdapter:
    def __init__(
        self,
        project_root: Path,
        query_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.project_root = project_root
        self._query_fn = query_fn

    def query(self, defect: str, observations: dict[str, Any]) -> dict[str, Any]:
        if self._query_fn is None:
            source = str(self.project_root / "kg/src")
            if source not in sys.path:
                sys.path.insert(0, source)
            from pcba_kg.query import query_causal_paths

            self._query_fn = query_causal_paths
        return self._query_fn(defect, observations=observations)
