"""Graph-to-text planning and deterministic realization for story graphs.

The realizer enforces exact event coverage and preserves role assignments.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Callable

from abstractgraph_generative.story.validation import validate_story_graph

EVENT_VERB = {
    "MOVE": "moved",
    "PREMISE": "was in a difficult situation with",
    "APPROACH": "approached",
    "LEAVE": "left",
    "ENTER": "entered",
    "EXIT": "exited",
    "SEARCH": "searched for",
    "FIND": "found",
    "LOSE": "lost",
    "SEE": "saw",
    "HEAR": "heard",
    "SPEAK": "spoke to",
    "ASK": "asked",
    "ANSWER": "answered",
    "WARN": "warned",
    "PROMISE": "promised",
    "REQUEST": "requested",
    "REFUSE": "refused",
    "AGREE": "agreed with",
    "THINK": "thought about",
    "BELIEVE": "believed",
    "DECIDE": "decided",
    "PLAN": "planned",
    "ATTEMPT": "attempted",
    "HELP": "helped",
    "TRICK": "tricked",
    "DECEIVE": "deceived",
    "TRUST": "trusted",
    "TAKE": "took",
    "GIVE": "gave",
    "STEAL": "stole",
    "DROP": "dropped",
    "HIDE": "hid",
    "OPEN": "opened",
    "CLOSE": "closed",
    "BUILD": "built",
    "BREAK": "broke",
    "THROW": "threw",
    "CATCH": "caught",
    "CHASE": "chased",
    "ESCAPE": "escaped from",
    "ATTACK": "attacked",
    "DEFEND": "defended",
    "INJURE": "injured",
    "KILL": "killed",
    "EAT": "ate",
    "DRINK": "drank",
    "WAIT": "waited for",
    "MEET": "met",
    "JOIN": "joined",
    "SEPARATE": "separated from",
    "PRAISE": "praised",
    "BLAME": "blamed",
    "JUDGE": "judged",
    "PUNISH": "punished",
    "REWARD": "rewarded",
    "DEMAND_TASK": "demanded a task from",
    "ASSIGN_TASK": "assigned a task to",
    "ACCEPT_TASK": "accepted a task from",
    "REFUSE_TASK": "refused a task from",
    "FULFILL_TASK": "fulfilled the task for",
    "DEMAND_PAYMENT": "demanded payment from",
    "PROMISE_PAYMENT": "promised payment to",
    "DENY_PAYMENT": "denied payment to",
    "BETRAY": "betrayed",
    "MAKE_DEAL": "made a deal with",
    "BREAK_PROMISE": "broke a promise to",
    "CURSE": "cursed",
    "BLESS": "blessed",
    "TRANSFORM": "transformed",
    "DISGUISE": "disguised",
    "REVEAL_IDENTITY": "revealed identity to",
    "TEST": "tested",
    "PASS_TEST": "passed the test set by",
    "FAIL_TEST": "failed the test set by",
    "BANISH": "banished",
    "RETURN_HOME": "returned home to",
    "MARRY": "married",
    "OTHER_EVENT": "did something to",
}

EVENT_VERB_INFINITIVE = {
    "MOVE": "move",
    "PREMISE": "be in a difficult situation with",
    "APPROACH": "approach",
    "LEAVE": "leave",
    "ENTER": "enter",
    "EXIT": "exit",
    "SEARCH": "search for",
    "FIND": "find",
    "LOSE": "lose",
    "SEE": "see",
    "HEAR": "hear",
    "SPEAK": "speak to",
    "ASK": "ask",
    "ANSWER": "answer",
    "WARN": "warn",
    "PROMISE": "promise",
    "REQUEST": "request",
    "REFUSE": "refuse",
    "AGREE": "agree with",
    "THINK": "think about",
    "BELIEVE": "believe",
    "DECIDE": "decide",
    "PLAN": "plan",
    "ATTEMPT": "attempt",
    "HELP": "help",
    "TRICK": "trick",
    "DECEIVE": "deceive",
    "TRUST": "trust",
    "TAKE": "take",
    "GIVE": "give",
    "STEAL": "steal",
    "DROP": "drop",
    "HIDE": "hide",
    "OPEN": "open",
    "CLOSE": "close",
    "BUILD": "build",
    "BREAK": "break",
    "THROW": "throw",
    "CATCH": "catch",
    "CHASE": "chase",
    "ESCAPE": "escape from",
    "ATTACK": "attack",
    "DEFEND": "defend",
    "INJURE": "injure",
    "KILL": "kill",
    "EAT": "eat",
    "DRINK": "drink",
    "WAIT": "wait for",
    "MEET": "meet",
    "JOIN": "join",
    "SEPARATE": "separate from",
    "PRAISE": "praise",
    "BLAME": "blame",
    "JUDGE": "judge",
    "PUNISH": "punish",
    "REWARD": "reward",
    "DEMAND_TASK": "demand a task from",
    "ASSIGN_TASK": "assign a task to",
    "ACCEPT_TASK": "accept a task from",
    "REFUSE_TASK": "refuse a task from",
    "FULFILL_TASK": "fulfill the task for",
    "DEMAND_PAYMENT": "demand payment from",
    "PROMISE_PAYMENT": "promise payment to",
    "DENY_PAYMENT": "deny payment to",
    "BETRAY": "betray",
    "MAKE_DEAL": "make a deal with",
    "BREAK_PROMISE": "break a promise to",
    "CURSE": "curse",
    "BLESS": "bless",
    "TRANSFORM": "transform",
    "DISGUISE": "disguise",
    "REVEAL_IDENTITY": "reveal identity to",
    "TEST": "test",
    "PASS_TEST": "pass the test set by",
    "FAIL_TEST": "fail the test set by",
    "BANISH": "banish",
    "RETURN_HOME": "return home to",
    "MARRY": "marry",
    "OTHER_EVENT": "act toward",
}

TRANSITIVE_EVENT_TYPES = {
    "SEARCH",
    "FIND",
    "LOSE",
    "SPEAK",
    "ASK",
    "ANSWER",
    "WARN",
    "PROMISE",
    "REQUEST",
    "REFUSE",
    "AGREE",
    "HELP",
    "TRICK",
    "DECEIVE",
    "TRUST",
    "TAKE",
    "GIVE",
    "STEAL",
    "DROP",
    "HIDE",
    "OPEN",
    "CLOSE",
    "BUILD",
    "BREAK",
    "THROW",
    "CATCH",
    "CHASE",
    "ESCAPE",
    "ATTACK",
    "DEFEND",
    "INJURE",
    "KILL",
    "EAT",
    "DRINK",
    "PREMISE",
    "MEET",
    "JOIN",
    "SEPARATE",
    "PRAISE",
    "BLAME",
    "JUDGE",
    "PUNISH",
    "REWARD",
    "DEMAND_TASK",
    "ASSIGN_TASK",
    "ACCEPT_TASK",
    "REFUSE_TASK",
    "FULFILL_TASK",
    "DEMAND_PAYMENT",
    "PROMISE_PAYMENT",
    "DENY_PAYMENT",
    "BETRAY",
    "MAKE_DEAL",
    "BREAK_PROMISE",
    "CURSE",
    "BLESS",
    "TRANSFORM",
    "DISGUISE",
    "REVEAL_IDENTITY",
    "TEST",
    "PASS_TEST",
    "FAIL_TEST",
    "BANISH",
    "RETURN_HOME",
    "MARRY",
    "OTHER_EVENT",
}


def _normalize_sentence_text(text: str) -> str:
    """Normalize sentence casing/punctuation for realization insertion.

    Args:
        text: Raw sentence-like text.

    Returns:
        Cleaned sentence ending with a period.
    """

    value = " ".join((text or "").strip().split())
    if not value:
        return ""
    value = value[0].upper() + value[1:] if len(value) > 1 else value.upper()
    if value[-1] not in ".!?":
        value += "."
    return value


def _naturalize_grounded_event_text(text: str, event_type: str) -> str:
    """Convert meta summary wording into direct narrative prose.

    Args:
        text: Grounded summary/evidence text.
        event_type: Event type label for prefix cleanup.

    Returns:
        Naturalized sentence text.
    """

    value = (text or "").strip()
    if not value:
        return ""

    # Remove schema-like prefixes such as "HEAR is the event where ..."
    prefixes = [
        rf"^\s*{re.escape(str(event_type).upper())}\s+is\s+the\s+event\s+where\s+",
        r"^\s*OTHER_EVENT\s+refers\s+to\s+",
        r"^\s*OTHER_EVENT\s+signifies\s+",
        r"^\s*[A-Z_]+\s+refers\s+to\s+",
        r"^\s*[A-Z_]+\s+signifies\s+",
        r"^\s*Intended action\s*\(not yet completed\)\s*:\s*",
        r"^\s*Obligated/expected action\s*:\s*",
    ]
    for pattern in prefixes:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)

    # Convert possessive nominalizations into finite verbs.
    value = re.sub(
        r"^\s*the\s+([A-Za-z][A-Za-z' \-]+?)'s\s+decision\s+to\s+",
        r"The \1 decides to ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*the\s+([A-Za-z][A-Za-z' \-]+?)'s\s+intention\s+to\s+",
        r"The \1 intends to ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^\s*the\s+decision\s+to\s+", "Decides to ", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*the\s+intention\s+to\s+", "Intends to ", value, flags=re.IGNORECASE)

    return _normalize_sentence_text(value)


def _node_name_in_sentence(node: dict | None, sentence: str) -> bool:
    """Return True if entity canonical name appears in sentence text."""

    if not node:
        return False
    name = str(node.get("canonical_name", "")).strip().lower()
    return bool(name) and name in sentence.lower()


def _select_primary_object_id(event_type: str, roles: dict[str, str], node_by_id: dict[str, dict]) -> str | None:
    """Select the best object role for realization based on event semantics.

    Args:
        event_type: Event type label.
        roles: Role mapping for one event.
        node_by_id: Node lookup.

    Returns:
        Chosen entity node id or None.
    """

    default_order = ["PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION"]
    role_priority = {
        "HEAR": ["TARGET", "PATIENT", "RECIPIENT"],
        "SEE": ["TARGET", "PATIENT", "RECIPIENT"],
        "SPEAK": ["TARGET", "RECIPIENT", "PATIENT"],
        "ASK": ["TARGET", "RECIPIENT", "PATIENT"],
        "ANSWER": ["TARGET", "RECIPIENT", "PATIENT"],
        "WARN": ["TARGET", "PATIENT", "RECIPIENT"],
        "BLAME": ["TARGET", "PATIENT", "RECIPIENT"],
        "PRAISE": ["TARGET", "PATIENT", "RECIPIENT"],
        "GIVE": ["RECIPIENT", "PATIENT", "TARGET"],
        "REQUEST": ["RECIPIENT", "PATIENT", "TARGET"],
        "PROMISE": ["RECIPIENT", "PATIENT", "TARGET"],
    }
    order = role_priority.get(str(event_type).upper(), default_order)
    chosen = next((roles.get(role) for role in order if roles.get(role)), None)

    # Avoid awkward sensory outputs like "heard the food" if an explicit target exists.
    if chosen and roles.get("TARGET"):
        chosen_node = node_by_id.get(chosen, {})
        if chosen_node.get("entity_type") in {"FOOD", "RESOURCE", "ABSTRACT"}:
            chosen = roles.get("TARGET")
    return chosen


def _entity_surface(entity_node: dict, first_mention_done: set[str]) -> str:
    """Resolve deterministic entity mention form.

    Args:
        entity_node: Entity node payload.
        first_mention_done: Set of entity IDs already mentioned.

    Returns:
        Surface string for realization.
    """

    entity_id = entity_node["id"]
    canonical = str(entity_node.get("canonical_name", entity_id)).strip() or entity_id
    label = canonical.lower()
    if entity_id not in first_mention_done:
        first_mention_done.add(entity_id)
        return f"the {label}"
    return f"the {label}"


def _event_order(graph: dict) -> list[str]:
    """Compute event order from BEFORE edges with deterministic tie-breaks.

    Args:
        graph: Validated story graph.

    Returns:
        Ordered event node IDs.
    """

    event_nodes = [node for node in graph["nodes"] if node.get("type") == "EVT"]
    node_by_id = {node["id"]: node for node in event_nodes}

    successors: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = {node["id"]: 0 for node in event_nodes}

    for edge in graph["edges"]:
        if edge.get("label") != "BEFORE":
            continue
        src = edge["source"]
        dst = edge["target"]
        if src not in indegree or dst not in indegree:
            continue
        if dst in successors[src]:
            continue
        successors[src].add(dst)
        indegree[dst] += 1

    ready = [node_id for node_id, degree in indegree.items() if degree == 0]

    def sort_key(node_id: str) -> tuple[str, str]:
        sentence_id = str(node_by_id[node_id]["provenance"].get("sentence_id", ""))
        return (sentence_id, node_id)

    ready.sort(key=sort_key)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for nxt in sorted(successors.get(current, set()), key=sort_key):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=sort_key)

    if len(ordered) != len(event_nodes):
        fallback = sorted([node["id"] for node in event_nodes], key=sort_key)
        return fallback

    return ordered


def _resolve_dictionary_dir(graph: dict, dictionary_dir: str | None = None) -> str | None:
    """Resolve dictionary directory from argument or graph metadata."""

    if dictionary_dir is not None:
        return dictionary_dir
    meta = graph.get("_dictionary_dir")
    if isinstance(meta, str) and meta.strip():
        return meta
    return None


def build_story_plan(graph: dict, *, dictionary_dir: str | None = None) -> list[dict]:
    """Linearize graph into an ordered event plan.

    Args:
        graph: Input story graph.
        dictionary_dir: Optional dictionary directory for induced vocab validation.

    Returns:
        Ordered list of plan records.
    """

    valid = validate_story_graph(graph, dictionary_dir=_resolve_dictionary_dir(graph, dictionary_dir))
    node_by_id = {node["id"]: node for node in valid["nodes"]}

    roles_by_event: dict[str, dict[str, str]] = defaultdict(dict)
    motives_by_event: dict[str, list[str]] = defaultdict(list)
    intention_by_id = {node["id"]: node for node in valid["nodes"] if node.get("type") == "INT"}
    intention_owner_by_id: dict[str, str] = {}
    intention_effects_by_event: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for edge in valid["edges"]:
        src = edge["source"]
        dst = edge["target"]
        label = edge["label"]
        if label in {"AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT", "LOCATION", "TIME"}:
            roles_by_event[src][label] = dst
        if label in {"MOTIVATES", "INTENDS", "HAS_GOAL", "FORMS_INTENTION"}:
            confidence = float(edge.get("confidence", 0.0))
            if confidence < 0.5:
                continue
            node = node_by_id.get(dst)
            if node and node.get("type") in {"GOAL", "INT"}:
                motives_by_event[src].append(str(node.get("goal_type", "OTHER_GOAL")))
        if label == "INTENDS" and src in node_by_id and dst in intention_by_id:
            intention_owner_by_id[dst] = src
        if label in {"ADVANCES", "THWARTS", "FULFILLS", "FAILS"} and dst in intention_by_id:
            owner = intention_owner_by_id.get(dst)
            intent_type = str(intention_by_id[dst].get("intention_type", "OTHER_INTENTION"))
            if owner:
                intention_effects_by_event[src].append((label, owner, intent_type))
            else:
                intention_effects_by_event[src].append((label, "", intent_type))

    plan: list[dict] = []
    for event_id in _event_order(valid):
        event_node = node_by_id[event_id]
        plan.append(
            {
                "order": len(plan) + 1,
                "event_id": event_id,
                "event_type": event_node.get("event_type", "OTHER_EVENT"),
                "modality": event_node.get("modality", "ASSERTED"),
                "summary_text": event_node.get("summary_text", ""),
                "evidence_text": event_node.get("evidence_text", ""),
                "sentence_id": event_node["provenance"].get("sentence_id", ""),
                "roles": dict(roles_by_event.get(event_id, {})),
                "motives": list(motives_by_event.get(event_id, [])),
                "intention_effects": list(intention_effects_by_event.get(event_id, [])),
            }
        )
    _validate_story_plan(plan=plan, node_by_id=node_by_id)
    return plan


def _validate_story_plan(plan: list[dict], node_by_id: dict[str, dict]) -> None:
    """Validate ordered story plan for sequencing and role completeness.

    Args:
        plan: Plan records.
        node_by_id: Graph node dictionary.

    Returns:
        None.
    """

    seen_events = set()
    for expected_order, record in enumerate(plan, start=1):
        if int(record.get("order", -1)) != expected_order:
            raise ValueError(f"Plan order mismatch at position {expected_order}: {record}")
        event_id = record.get("event_id")
        if event_id in seen_events:
            raise ValueError(f"Duplicate event in plan: {event_id}")
        seen_events.add(event_id)

        roles = dict(record.get("roles", {}))
        if "AGENT" not in roles:
            raise ValueError(f"Plan event {event_id} missing AGENT role")

        # Duplicate semantic signatures can legitimately occur in stories
        # (repeated attempts/actions), so we only enforce ordering and role presence.


def realize_story_from_graph(
    graph: dict,
    *,
    dictionary_dir: str | None = None,
) -> tuple[str, list[dict]]:
    """Realize a story text from a graph with exact event coverage.

    Args:
        graph: Input story graph.
        dictionary_dir: Optional dictionary directory for induced vocab validation.

    Returns:
        Tuple `(story_text, plan_records)`.
    """

    resolved_dictionary_dir = _resolve_dictionary_dir(graph, dictionary_dir)
    valid = validate_story_graph(graph, dictionary_dir=resolved_dictionary_dir)
    plan = build_story_plan(valid, dictionary_dir=resolved_dictionary_dir)

    node_by_id = {node["id"]: node for node in valid["nodes"]}
    first_mention_done: set[str] = set()

    lines: list[str] = []
    for record in plan:
        event_type = record["event_type"]
        modality = str(record.get("modality", "ASSERTED")).upper()
        verb = EVENT_VERB.get(event_type, "did something to")
        inf_verb = EVENT_VERB_INFINITIVE.get(event_type, "act toward")
        summary_text = str(record.get("summary_text", "")).strip()
        evidence_text = str(record.get("evidence_text", "")).strip()

        roles = record["roles"]
        agent_node = node_by_id.get(roles.get("AGENT", ""))
        object_id = _select_primary_object_id(event_type=event_type, roles=roles, node_by_id=node_by_id)
        patient_node = node_by_id.get(object_id or "")

        if agent_node is None:
            raise ValueError(f"Event {record['event_id']} missing AGENT during realization")

        grounded_raw = summary_text or evidence_text
        grounded = _naturalize_grounded_event_text(grounded_raw, event_type)
        prefers_grounded = event_type in {
            "PREMISE",
            "HEAR",
            "SEE",
            "SPEAK",
            "ASK",
            "ANSWER",
            "BLAME",
            "PRAISE",
            "DECEIVE",
            "PROMISE",
            "PROMISE_PAYMENT",
            "DEMAND_PAYMENT",
            "DENY_PAYMENT",
            "REWARD",
            "FULFILL_TASK",
            "DEMAND_TASK",
            "ASSIGN_TASK",
            "ACCEPT_TASK",
            "REFUSE_TASK",
            "MAKE_DEAL",
            "BREAK_PROMISE",
            "CURSE",
            "BLESS",
            "TRANSFORM",
            "DISGUISE",
            "REVEAL_IDENTITY",
            "TEST",
            "PASS_TEST",
            "FAIL_TEST",
            "BANISH",
            "RETURN_HOME",
            "MARRY",
            "BETRAY",
        }

        # Prefer grounded text for vague event typing or role-incomplete transitive events.
        if (
            event_type == "OTHER_EVENT"
            or (event_type in TRANSITIVE_EVENT_TYPES and patient_node is None)
            or (prefers_grounded and grounded)
        ):
            if grounded:
                sentence = grounded
                if not _node_name_in_sentence(agent_node, sentence):
                    agent_txt = _entity_surface(agent_node, first_mention_done)
                    sentence = _normalize_sentence_text(f"{agent_txt} {sentence[0].lower() + sentence[1:]}")
            else:
                agent_txt = _entity_surface(agent_node, first_mention_done)
                if modality == "POSSIBLE":
                    sentence = f"{agent_txt} intended to act."
                elif modality == "OBLIGATED":
                    sentence = f"{agent_txt} was expected to act."
                else:
                    sentence = f"{agent_txt} acted."
        else:
            agent_txt = _entity_surface(agent_node, first_mention_done)
            if patient_node is not None:
                patient_txt = _entity_surface(patient_node, first_mention_done)
                if modality == "POSSIBLE":
                    sentence = f"{agent_txt} intended to {inf_verb} {patient_txt}."
                elif modality == "OBLIGATED":
                    sentence = f"{agent_txt} was expected to {inf_verb} {patient_txt}."
                else:
                    sentence = f"{agent_txt} {verb} {patient_txt}."
            else:
                if modality == "POSSIBLE":
                    sentence = f"{agent_txt} intended to act."
                elif modality == "OBLIGATED":
                    sentence = f"{agent_txt} was expected to act."
                else:
                    sentence = f"{agent_txt} {verb}."

        if record["motives"]:
            motive = record["motives"][0].lower().replace("_", " ")
            sentence = sentence[:-1] + f" to {motive}."

        if record.get("intention_effects"):
            effect, owner_id, intent_type = record["intention_effects"][0]
            owner_node = node_by_id.get(owner_id) if owner_id else None
            owner_txt = _entity_surface(owner_node, first_mention_done) if owner_node else "the plan"
            intent_txt = intent_type.lower().replace("_", " ")
            if effect == "ADVANCES":
                sentence = sentence[:-1] + f", which advanced {owner_txt}'s intention to {intent_txt}."
            elif effect == "THWARTS":
                sentence = sentence[:-1] + f", which hindered {owner_txt}'s intention to {intent_txt}."
            elif effect == "FULFILLS":
                sentence = sentence[:-1] + f", which fulfilled {owner_txt}'s intention to {intent_txt}."
            elif effect == "FAILS":
                sentence = sentence[:-1] + f", and {owner_txt}'s intention to {intent_txt} failed."

        lines.append(sentence[0].upper() + sentence[1:])

    story = " ".join(lines).strip()
    verify_event_coverage(valid, plan)
    return story, plan


def _plan_signature(plan: list[dict], graph: dict) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    """Create normalized plan signature for consistency checks.

    Args:
        plan: Plan records.
        graph: Source graph.

    Returns:
        Ordered signature list.
    """

    node_by_id = {node["id"]: node for node in graph["nodes"]}
    signature = []
    for row in plan:
        role_pairs = []
        for role, ent_id in sorted(row.get("roles", {}).items()):
            ent = node_by_id.get(ent_id, {})
            role_pairs.append((role, str(ent.get("canonical_name", ent_id)).lower()))
        signature.append((row.get("event_type", "OTHER_EVENT"), tuple(role_pairs)))
    return signature


def _normalize_event_type_for_match(event_type: str) -> str:
    """Normalize event type for robust repair matching."""

    value = str(event_type or "OTHER_EVENT").upper()
    if value in {"DECIDE", "PLAN", "THINK", "BELIEVE", "ATTEMPT"}:
        return "COGNITIVE_EVENT"
    return value


def _signatures_compatible(
    left: list[tuple[str, tuple[tuple[str, str], ...]]],
    right: list[tuple[str, tuple[tuple[str, str], ...]]],
) -> bool:
    """Return True if two signatures preserve order/agents under coarse typing."""

    if len(left) != len(right):
        return False
    for (lt, lroles), (rt, rroles) in zip(left, right):
        nlt = _normalize_event_type_for_match(lt)
        nrt = _normalize_event_type_for_match(rt)
        if nlt != nrt:
            return False

        ldict = {k: v for k, v in lroles}
        rdict = {k: v for k, v in rroles}
        for role in ("AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT"):
            lv = ldict.get(role, "")
            rv = rdict.get(role, "")
            if lv or rv:
                if lv != rv:
                    return False

        lcore = tuple((k, ldict.get(k, "")) for k in ("AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT"))
        rcore = tuple((k, rdict.get(k, "")) for k in ("AGENT", "PATIENT", "TARGET", "RECIPIENT", "INSTRUMENT"))
        if lcore != rcore:
            return False
    return True


def realize_story_from_graph_with_repair(
    graph: dict,
    *,
    ask_llm_fn: Callable[..., str],
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
    api_key: str | None = None,
    dictionary_dir: str | None = None,
) -> tuple[str, list[dict], dict]:
    """Realize story from graph and perform one sequencing repair pass.

    Args:
        graph: Input story graph.
        ask_llm_fn: LLM callable.
        model: Optional model name.
        temperature: Sampling temperature.
        max_output_tokens: Max output tokens.
        api_key: Optional API key override.
        dictionary_dir: Optional dictionary directory for vocab-dependent parsing.

    Returns:
        `(story_text, plan, diagnostics)` tuple.
    """

    base_story, plan = realize_story_from_graph(graph, dictionary_dir=dictionary_dir)
    from abstractgraph_generative.story.text_to_graph import build_story_graph_from_text

    original_sig = _plan_signature(plan, graph)
    regen_graph = build_story_graph_from_text(
        text=base_story,
        doc_id=f"{graph.get('doc_id', 'story')}_regen_check",
        ask_llm_fn=ask_llm_fn,
        model=model,
        api_key=api_key,
        dictionary_dir=dictionary_dir,
    )
    regen_plan = build_story_plan(regen_graph, dictionary_dir=dictionary_dir)
    regen_sig = _plan_signature(regen_plan, regen_graph)
    if _signatures_compatible(original_sig, regen_sig):
        return base_story, plan, {"repaired": False, "sequence_match": True}

    plan_lines = []
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    for row in plan:
        roles = row.get("roles", {})
        agent = node_by_id.get(roles.get("AGENT", ""), {}).get("canonical_name", "Unknown")
        patient = node_by_id.get(roles.get("PATIENT", ""), {}).get("canonical_name", "")
        summary_hint = str(row.get("summary_text", "")).strip()
        evidence_hint = str(row.get("evidence_text", "")).strip()
        if patient:
            role_txt = f"AGENT={agent}; PATIENT={patient}"
        else:
            role_txt = f"AGENT={agent}"
        hint_txt = summary_hint or evidence_hint
        if hint_txt:
            plan_lines.append(f"EVENT_{row['order']:02d}: TYPE={row['event_type']}; {role_txt}; HINT={hint_txt}")
        else:
            plan_lines.append(f"EVENT_{row['order']:02d}: TYPE={row['event_type']}; {role_txt}")

    repaired_story = base_story
    repaired_sig = regen_sig
    attempts = 0
    while attempts < 2 and not _signatures_compatible(original_sig, repaired_sig):
        attempts += 1
        repair_prompt = (
            "Rewrite the story so it matches this ordered event plan exactly.\n"
            "Hard constraints:\n"
            f"1) Use exactly {len(plan)} sentences, in this exact event order.\n"
            "2) Preserve event order strictly.\n"
            "3) Keep agent/patient roles unchanged.\n"
            "4) Do not add or remove events.\n"
            "5) Keep wording specific, avoid generic placeholders.\n\n"
            "6) Respect each event HINT when present; do not invert its meaning.\n"
            "7) Output plain prose only. No numbering, bullets, or labels.\n\n"
            "Event plan:\n"
            + "\n".join(plan_lines)
            + "\n\nCurrent story:\n"
            + repaired_story
            + "\n\nReturn only the repaired story."
        )
        repaired_story = ask_llm_fn(
            question=repair_prompt,
            model=model,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            system_prompt="You are a strict sequence-preserving story repair assistant.",
        ).strip()

        repaired_graph = build_story_graph_from_text(
            text=repaired_story,
            doc_id=f"{graph.get('doc_id', 'story')}_regen_repaired_{attempts}",
            ask_llm_fn=ask_llm_fn,
            model=model,
            api_key=api_key,
            add_node_summaries=False,
            dictionary_dir=dictionary_dir,
        )
        repaired_plan = build_story_plan(repaired_graph, dictionary_dir=dictionary_dir)
        repaired_sig = _plan_signature(repaired_plan, repaired_graph)

    return repaired_story, plan, {"repaired": True, "sequence_match": _signatures_compatible(original_sig, repaired_sig), "repair_attempts": attempts}


def verify_event_coverage(graph: dict, plan: list[dict]) -> None:
    """Ensure each event appears exactly once in the plan.

    Args:
        graph: Input graph.
        plan: Linearized plan records.

    Returns:
        None.
    """

    graph_event_ids = sorted(node["id"] for node in graph["nodes"] if node.get("type") == "EVT")
    plan_event_ids = sorted(record["event_id"] for record in plan)

    if graph_event_ids != plan_event_ids:
        raise ValueError(
            "Event coverage check failed: plan events do not match graph events. "
            f"graph={graph_event_ids}, plan={plan_event_ids}"
        )


def render_story_with_style(
    generated_story: str,
    style_prompt: str,
    *,
    ask_llm_fn: Callable[..., str],
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
    api_key: str | None = None,
) -> str:
    """Render a more detailed story while preserving generated-story content.

    Args:
        generated_story: Baseline story generated from graph realization.
        style_prompt: User-provided tone/style instruction.
        ask_llm_fn: Callable used to query an LLM.
        model: Optional model name.
        temperature: Sampling temperature.
        max_output_tokens: Maximum generated tokens.
        api_key: Optional API key override.

    Returns:
        Styled story text constrained to the same event/role content.
    """

    prompt = (
        "Rewrite the baseline story in richer detail and smoother prose while preserving exact story facts.\n"
        "Hard constraints:\n"
        "1) Keep the same events and event order.\n"
        "2) Keep the same actor/recipient roles for each event.\n"
        "3) Do not invent new events, characters, causes, or outcomes.\n"
        "4) Do not omit any baseline event.\n"
        "5) You may add sensory/contextual detail and phrasing only.\n\n"
        f"Style request:\n{style_prompt}\n\n"
        f"Baseline story:\n{generated_story}\n\n"
        "Return only the rewritten story text."
    )

    return ask_llm_fn(
        question=prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
        system_prompt=(
            "You are a faithful narrative rewriter. "
            "Never add or remove events; preserve role assignments exactly."
        ),
    ).strip()
