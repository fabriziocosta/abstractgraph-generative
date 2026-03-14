"""Dataset packaging helpers for text/graph corpora.

This module writes aligned story text files, graph JSON files, metadata, and
reproducible split manifests for training workflows.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from abstractgraph_generative.story.validation import validate_story_graph


def package_story_dataset(
    records: list[dict],
    out_dir: str | Path,
    *,
    seed: int = 13,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict:
    """Write a packaged story dataset to disk.

    Args:
        records: Items with keys `doc_id`, `title`, `source`, `text`, `graph`.
        out_dir: Output directory.
        seed: Random seed for deterministic split.
        train_frac: Train split fraction.
        val_frac: Validation split fraction.

    Returns:
        Dictionary with split sizes.
    """

    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("Invalid split fractions: require 0 < train_frac, 0 <= val_frac, train+val < 1")

    base = Path(out_dir)
    stories_dir = base / "stories"
    graphs_dir = base / "graphs"
    splits_dir = base / "splits"

    stories_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict] = []
    doc_ids: list[str] = []

    for record in records:
        doc_id = str(record["doc_id"])
        text = str(record["text"])
        graph = validate_story_graph(record["graph"])

        (stories_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        (graphs_dir / f"{doc_id}.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

        doc_ids.append(doc_id)
        metadata_rows.append(
            {
                "doc_id": doc_id,
                "title": str(record.get("title", "")),
                "source": str(record.get("source", "")),
                "length_chars": len(text),
                "length_words": len(text.split()),
            }
        )

    metadata_path = base / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["doc_id", "title", "source", "length_chars", "length_words"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    rng = random.Random(seed)
    shuffled = list(doc_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    (splits_dir / "train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""), encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""), encoding="utf-8")
    (splits_dir / "test.txt").write_text("\n".join(test_ids) + ("\n" if test_ids else ""), encoding="utf-8")

    return {
        "total": n_total,
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
        "out_dir": str(base),
    }
