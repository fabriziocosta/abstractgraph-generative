"""Smoke test for the extracted abstractgraph-generative package."""

from __future__ import annotations

from abstractgraph_generative.conditional import ConditionalAutoregressiveGenerator


def main() -> None:
    """Run a minimal generative import smoke test."""
    print("generator", ConditionalAutoregressiveGenerator.__name__)


if __name__ == "__main__":
    main()
