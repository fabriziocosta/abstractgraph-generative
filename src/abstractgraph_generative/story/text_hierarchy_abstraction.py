"""WordNet-based abstraction utilities for event text simplification."""

from __future__ import annotations

import nltk
import spacy
from nltk.corpus import wordnet as wn

_VOCAB_NLP = None


_TECHNICAL_HYPERNYM_LEMMAS = {
    "felid",
    "canid",
    "hominid",
    "bovid",
    "equid",
    "carnivore",
    "placental",
    "vertebrate",
}


def get_vocab_nlp(model_name: str = "en_core_web_sm"):
    """Return a cached spaCy pipeline for lemma/POS checks.

    Args:
        model_name: spaCy model name.

    Returns:
        Loaded spaCy language pipeline.
    """

    global _VOCAB_NLP
    if _VOCAB_NLP is None:
        _VOCAB_NLP = spacy.load(model_name, disable=["parser", "ner", "textcat"])
    return _VOCAB_NLP


def ensure_wordnet_resources() -> bool:
    """Ensure WordNet corpora are available.

    Args:
        None.

    Returns:
        True if WordNet lookups are available, else False.
    """

    try:
        _ = wn.synsets("dog")
        return True
    except LookupError:
        pass

    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        _ = wn.synsets("dog")
        return True
    except Exception:
        return False


def spacy_pos_to_wordnet_pos(pos: str):
    """Map spaCy POS tags to WordNet POS tags.

    Args:
        pos: spaCy part-of-speech tag.

    Returns:
        WordNet POS tag or None.
    """

    if pos in {"NOUN", "PROPN"}:
        return wn.NOUN
    if pos == "VERB":
        return wn.VERB
    if pos == "ADJ":
        return wn.ADJ
    if pos == "ADV":
        return wn.ADV
    return None


def preferred_synset_for_lemma(lemma: str, wn_pos):
    """Select a stable synset for a lemma with deterministic scoring.

    Args:
        lemma: Input lemma.
        wn_pos: WordNet POS tag.

    Returns:
        Best matching synset or None.
    """

    synsets = wn.synsets(lemma, pos=wn_pos)
    if not synsets:
        return None

    target = lemma.replace(" ", "_").lower()

    def score(syn):
        lemmas = syn.lemmas()
        exact = max((l.count() for l in lemmas if l.name().lower() == target), default=0)
        first_exact = 1 if lemmas and lemmas[0].name().lower() == target else 0
        total = sum(l.count() for l in lemmas)
        return (exact, first_exact, total)

    return sorted(synsets, key=lambda syn: (-score(syn)[0], -score(syn)[1], -score(syn)[2], syn.name()))[0]


def best_hypernym_lemma(lemma: str, wn_pos, n_up_levels: int) -> str:
    """Lift one lemma through WordNet hypernyms deterministically.

    Args:
        lemma: Input lemma.
        wn_pos: WordNet POS tag.
        n_up_levels: Number of hypernym hops.

    Returns:
        Hypernym lemma after climbing levels, or original lemma if unavailable.
    """

    if not lemma or not wn_pos or n_up_levels < 1:
        return lemma

    syn = preferred_synset_for_lemma(lemma=lemma, wn_pos=wn_pos)
    if syn is None:
        return lemma

    for _ in range(n_up_levels):
        hypers = syn.hypernyms()
        if not hypers:
            break
        syn = sorted(hypers, key=lambda item: (-sum(l.count() for l in item.lemmas()), item.name()))[0]

    lemma_candidates: list[tuple[int, int, str]] = []
    for l in syn.lemmas():
        name = l.name().replace("_", " ").lower().strip()
        if not name:
            continue
        penalty = 0
        if name in _TECHNICAL_HYPERNYM_LEMMAS:
            penalty += 3
        if " " in name:
            penalty += 1
        lemma_candidates.append((penalty, -l.count(), name))

    if not lemma_candidates:
        return lemma

    return sorted(lemma_candidates)[0][2]


def wordnet_hypernym_simplify_event(event: str, n_up_levels: int, nlp) -> str:
    """Rewrite one event by replacing content-word lemmas with hypernyms.

    Args:
        event: Event text.
        n_up_levels: Number of hypernym hops.
        nlp: spaCy pipeline for tokenization/POS/lemma.

    Returns:
        Simplified event text.
    """

    doc = nlp(event or "")
    out: list[str] = []

    for tok in doc:
        text = tok.text
        if tok.is_alpha:
            lemma = tok.lemma_.strip().lower()
            if lemma == "-pron-":
                lemma = tok.lower_
            wn_pos = spacy_pos_to_wordnet_pos(tok.pos_)
            if wn_pos:
                hyper = best_hypernym_lemma(lemma=lemma, wn_pos=wn_pos, n_up_levels=n_up_levels)
                if hyper and hyper not in _TECHNICAL_HYPERNYM_LEMMAS:
                    text = hyper.capitalize() if tok.text[:1].isupper() else hyper

        out.append(text + tok.whitespace_)

    return "".join(out).strip()


def simplify_events_with_wordnet(items: list[str], n_up_levels: int = 2) -> list[str]:
    """Simplify events by replacing words with WordNet hypernyms.

    Args:
        items: Event strings to simplify.
        n_up_levels: Number of hypernym levels to climb per content word.

    Returns:
        Simplified event strings.
    """

    if not items:
        return items
    if n_up_levels < 1:
        return list(items)
    if not ensure_wordnet_resources():
        return list(items)

    nlp = get_vocab_nlp()
    return [wordnet_hypernym_simplify_event(event=it, n_up_levels=n_up_levels, nlp=nlp) for it in items]
