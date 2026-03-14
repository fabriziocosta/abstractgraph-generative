"""Types for conditional generative workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx


@dataclass(frozen=True)
class CutArchiveEntry:
    """Container for donor association data used in anchor-merge insertion."""

    assoc: nx.Graph
    assoc_hash: int
    image_node: Optional[object] = None
    image_label: Optional[object] = None
    anchor_nodes: tuple = ()
    anchor_outer_hashes: tuple = ()
    anchor_inner_hashes: tuple = ()
    anchor_pairs: tuple = ()
    neighbor_overlap_signature: tuple = ()
    cut_hash: int = 0
    preimage_ctx: Optional[object] = None
    image_ctx: Optional[object] = None
    source: str = "pruning"
