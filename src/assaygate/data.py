"""Dataset loading (JSONL or CSV) into ``Item`` records."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Item(BaseModel):
    id: str
    input: Any
    expected: Any = None
    fields: dict[str, Any] = Field(default_factory=dict, description="the full original row")


def load_rows(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".csv":
        with path.open(newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"unsupported dataset format {path.suffix!r} (use .jsonl or .csv)")


def load_items(
    path: Path | str,
    *,
    id_field: str = "id",
    input_field: str = "input",
    expected_field: str | None = "expected",
    limit: int | None = None,
) -> list[Item]:
    rows = load_rows(path)
    if limit is not None:
        rows = rows[:limit]
    items: list[Item] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if input_field not in row:
            raise KeyError(f"row {i} has no {input_field!r} field")
        item_id = str(row.get(id_field, i))
        if item_id in seen:
            raise ValueError(f"duplicate item id {item_id!r}: ids must be unique to pair runs")
        seen.add(item_id)
        items.append(
            Item(
                id=item_id,
                input=row[input_field],
                expected=row.get(expected_field) if expected_field else None,
                fields=row,
            )
        )
    return items


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
