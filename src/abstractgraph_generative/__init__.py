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
from abstractgraph_generative.edge_generator import (  # noqa: F401
    EdgeGenerator,
    edge_neighbors,
    make_edge_regression_dataset,
    remove_edges,
)
try:
    from abstractgraph_generative.vgae_networkx_wrapper import (  # noqa: F401
        VGAENetworkXGenerator,
    )
except ModuleNotFoundError:
    VGAENetworkXGenerator = None
_story_import_error = None
try:
    from abstractgraph_generative.story import (  # noqa: F401
        SCHEMA_VERSION as STORY_GRAPH_SCHEMA_VERSION,
        StoryGraphValidationError,
        build_story_graph_from_text,
        build_story_plan,
        graph_validity_stats,
        ask_llm as story_graph_ask_llm,
        call_openai_llm as story_graph_call_openai_llm,
        dump_json as story_graph_dump_json,
        extract_story_graph as story_graph_extract_story_graph,
        generate_story_with_repair as story_graph_generate_story_with_repair,
        generate_styled_story as story_graph_generate_styled_story,
        intention_summary_rows as story_graph_intention_summary_rows,
        load_aesop_fables_from_gutenberg,
        load_vocab as load_story_graph_vocab,
        package_story_dataset,
        plot_story_graph,
        render_story_graph_dot,
        render_story_graph_dot_svg as story_graph_render_story_graph_dot_svg,
        render_story_with_style,
        realize_story_from_graph,
        realize_story_from_graph_with_repair,
        resolve_model_name as story_graph_resolve_model_name,
        roundtrip_check as story_graph_roundtrip_check,
        select_story as story_graph_select_story,
        story_graph_to_dot_string,
        to_networkx_story_graph,
        validate_story_graph,
        verify_event_coverage,
        write_provenance_audit_html,
    )
except (ImportError, ModuleNotFoundError) as exc:
    _story_import_error = exc

_STORY_EXPORTS = [
    "STORY_GRAPH_SCHEMA_VERSION",
    "StoryGraphValidationError",
    "build_story_graph_from_text",
    "validate_story_graph",
    "build_story_plan",
    "realize_story_from_graph",
    "realize_story_from_graph_with_repair",
    "render_story_with_style",
    "verify_event_coverage",
    "graph_validity_stats",
    "load_aesop_fables_from_gutenberg",
    "story_graph_call_openai_llm",
    "story_graph_ask_llm",
    "story_graph_resolve_model_name",
    "story_graph_select_story",
    "story_graph_extract_story_graph",
    "story_graph_intention_summary_rows",
    "story_graph_render_story_graph_dot_svg",
    "story_graph_generate_story_with_repair",
    "story_graph_generate_styled_story",
    "story_graph_roundtrip_check",
    "story_graph_dump_json",
    "plot_story_graph",
    "story_graph_to_dot_string",
    "render_story_graph_dot",
    "to_networkx_story_graph",
    "write_provenance_audit_html",
    "package_story_dataset",
    "load_story_graph_vocab",
]
__all__ = [
    "ConditionalAutoregressiveGenerator",
    "AttributedConditionalAutoregressiveGenerator",
    "ConditionalAutoregressiveGraphsGenerator",
    "GraphOptimizationResult",
    "GraphOptimizer",
    "VGAEGraphGenerator",
    "GRANGraphGenerator",
    "DiGressGraphGenerator",
    "EdgeGenerator",
    "edge_neighbors",
    "make_edge_regression_dataset",
    "remove_edges",
]

if _story_import_error is None:
    __all__.extend(_STORY_EXPORTS)

if VGAENetworkXGenerator is not None:
    __all__.append("VGAENetworkXGenerator")


def __getattr__(name: str):
    if name in _STORY_EXPORTS and _story_import_error is not None:
        raise ImportError(
            "Story utilities are unavailable in this checkout because the story package could not be imported."
        ) from _story_import_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
