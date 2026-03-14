"""Story-graph extraction/realization package with strict discrete schema."""

from __future__ import annotations

from abstractgraph_generative.story.audit import graph_validity_stats, write_provenance_audit_html
from abstractgraph_generative.story.dataset import package_story_dataset
from abstractgraph_generative.story.graph_to_text import (
    build_story_plan,
    realize_story_from_graph_with_repair,
    render_story_with_style,
    realize_story_from_graph,
    verify_event_coverage,
)
from abstractgraph_generative.story.lexicon_induction import (
    build_raw_term_inventory,
    consolidate_category_terms,
    extract_story_term_candidates,
    induce_closed_alphabets,
    load_induced_alphabets,
    save_induced_alphabets,
)
from abstractgraph_generative.story.notebook_utils import (
    dictionary_usage_stats,
    dump_json,
    entity_summary_rows,
    event_summary_rows,
    extract_story_graph,
    generate_story_with_repair,
    generate_styled_story,
    goal_summary_rows,
    intention_summary_rows,
    relation_summary_rows,
    print_graph_semantic_layers,
    render_story_graph_dot_svg,
    resolve_model_name,
    roundtrip_check,
    select_story,
)
from abstractgraph_generative.story.schema import SCHEMA_VERSION
from abstractgraph_generative.story.text_to_graph import build_story_graph_from_text
from abstractgraph_generative.story.utils import (
    ask_llm,
    call_openai_llm,
    format_text_max_columns,
    load_aesop_fables_from_gutenberg,
    pretty_print_text,
)
from abstractgraph_generative.story.validation import StoryGraphValidationError, validate_story_graph
from abstractgraph_generative.story.visualization import (
    plot_story_graph,
    render_story_graph_dot,
    story_graph_to_dot_string,
    to_networkx_story_graph,
)
from abstractgraph_generative.story.vocab import load_vocab

__all__ = [
    "SCHEMA_VERSION",
    "StoryGraphValidationError",
    "build_story_graph_from_text",
    "validate_story_graph",
    "build_story_plan",
    "realize_story_from_graph",
    "realize_story_from_graph_with_repair",
    "render_story_with_style",
    "verify_event_coverage",
    "graph_validity_stats",
    "write_provenance_audit_html",
    "package_story_dataset",
    "load_vocab",
    "load_aesop_fables_from_gutenberg",
    "format_text_max_columns",
    "pretty_print_text",
    "call_openai_llm",
    "ask_llm",
    "plot_story_graph",
    "story_graph_to_dot_string",
    "render_story_graph_dot",
    "to_networkx_story_graph",
    "resolve_model_name",
    "select_story",
    "extract_story_graph",
    "entity_summary_rows",
    "goal_summary_rows",
    "intention_summary_rows",
    "event_summary_rows",
    "relation_summary_rows",
    "print_graph_semantic_layers",
    "render_story_graph_dot_svg",
    "generate_story_with_repair",
    "generate_styled_story",
    "roundtrip_check",
    "dump_json",
    "dictionary_usage_stats",
    "extract_story_term_candidates",
    "build_raw_term_inventory",
    "consolidate_category_terms",
    "induce_closed_alphabets",
    "save_induced_alphabets",
    "load_induced_alphabets",
]
