"""Generative models and utilities for AbstractGraph."""

from __future__ import annotations

from abstractgraph_generative.conditional import (  # noqa: F401
    ConditionalAutoregressiveGenerator,
)
from abstractgraph_generative.conditional_attributed import (  # noqa: F401
    AttributedConditionalAutoregressiveGenerator,
)
from abstractgraph_generative.conditional_batch import (  # noqa: F401
    ConditionalAutoregressiveGraphsGenerator,
)
from abstractgraph_generative.optimize import (  # noqa: F401
    GraphOptimizationResult,
    GraphOptimizer,
)
from abstractgraph_generative.vgae_graph_generator import (  # noqa: F401
    VGAEGraphGenerator,
)
from abstractgraph_generative.gran_graph_generator import (  # noqa: F401
    GRANGraphGenerator,
)
from abstractgraph_generative.digress_graph_generator import (  # noqa: F401
    DiGressGraphGenerator,
)
try:
    from abstractgraph_generative.vgae_networkx_wrapper import (  # noqa: F401
        VGAENetworkXGenerator,
    )
except ModuleNotFoundError:
    VGAENetworkXGenerator = None
__all__ = [
    "ConditionalAutoregressiveGenerator",
    "AttributedConditionalAutoregressiveGenerator",
    "ConditionalAutoregressiveGraphsGenerator",
    "GraphOptimizationResult",
    "GraphOptimizer",
    "VGAEGraphGenerator",
    "GRANGraphGenerator",
    "DiGressGraphGenerator",
]

if VGAENetworkXGenerator is not None:
    __all__.append("VGAENetworkXGenerator")
