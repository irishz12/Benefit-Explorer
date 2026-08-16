"""Lightweight query signals used by the optional cross-encoder reranker."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

import numpy as np

from src.retrieval.embeddings import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SPECIFIC_QUERY = re.compile(
    r"\b(?:how\s+(?:is|are)|determined|calculated|payable\s+when|when\s+(?:is|are|does)|"
    r"what\s+happens|minimum|maximum|how\s+often)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "fifteen": "15",
    "thirty": "30",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "kotak",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "what",
    "when",
    "which",
    "with",
}

_ANSWER_VERB_PATTERNS = (
    "shall be paid",
    "shall be payable",
    "is paid",
    "is payable",
    "will be paid",
    "will be payable",
    "receive",
    "receives",
)

_MATURITY_ANCHORS = (
    "sum assured on maturity will be paid",
    "sum assured on maturity shall be paid",
    "sum assured on maturity plus",
    "fund value inclusive of loyalty additions shall be payable",
    "fund value is payable at maturity",
    "maturity benefit shall be payable",
    "maturity benefit will be paid",
)

_ANSWER_ANCHOR_INTENTS = frozenset(
    {
        "bonus_payout",
        "charge_return",
        "early_restriction",
        "income_start",
        "insta_cashback",
        "investment_strategy",
        "maturity_benefit",
        "named_feature",
        "option_count",
        "suicide_exclusion",
    }
)


@dataclass(frozen=True, slots=True)
class QueryIntent:
    name: str
    preferred_sections: tuple[str, ...]
    compatible_sections: tuple[str, ...]
    anchor_phrases: tuple[str, ...]
    specific: bool


@dataclass(frozen=True, slots=True)
class FocusedWindow:
    text: str
    score: float
    lexical_score: float
    embedding_score: float | None


@dataclass(frozen=True, slots=True)
class _IntentRule:
    name: str
    patterns: tuple[str, ...]
    preferred_sections: tuple[str, ...]
    compatible_sections: tuple[str, ...]
    anchor_phrases: tuple[str, ...]


_INTENT_RULES = (
    _IntentRule(
        "suicide_exclusion",
        (
            "suicide exclusion",
            "dies by suicide",
            "committing suicide",
            "suicide within",
            "suicide",
        ),
        ("Exclusions",),
        ("Benefits", "Policy Terms"),
        (
            "committing suicide within 12 months",
            "suicide within 12 months",
            "80% of total premiums paid",
            "death benefit under the product shall be payable",
        ),
    ),
    _IntentRule(
        "charge_return",
        (
            "return of premium allocation charge",
            "premium allocation charge",
            "return of mortality charges",
            "mortality charges",
            "return any premium allocation",
        ),
        ("Charges", "Benefits"),
        ("Policy Terms", "Premiums"),
        (
            "return of premium allocation charge",
            "premium allocation charge will be refunded",
            "return of mortality charges",
            "shall return multiple of mortality charges",
            "added back to the fund value",
        ),
    ),
    _IntentRule(
        "investment_strategy",
        (
            "investment strategy",
            "investment strategies",
            "self managed strategy",
            "age based strategy",
            "fund strategy",
        ),
        ("Benefits", "General"),
        ("Policy Terms",),
        (
            "choose from 2 investment strategies",
            "investment strategies",
            "self managed strategy",
            "age based strategy",
            "fund options",
        ),
    ),
    _IntentRule(
        "bonus_payout",
        (
            "bonuses be paid",
            "bonus paid",
            "bonus payout",
        ),
        ("Benefits",),
        ("General", "Policy Terms", "Premiums"),
        (
            "cash bonus immediate payout",
            "cash bonus deferred payout",
            "paid up additions",
        ),
    ),
    _IntentRule(
        "income_start",
        (
            "start paying",
            "start receiving",
            "income start",
            "income starts",
            "income begin",
            "income begins",
            "income commence",
            "income commences",
            "first guaranteed income",
        ),
        ("Benefits",),
        ("Policy Terms", "Premiums"),
        (
            "guaranteed income shall commence",
            "commence after the end of deferment period",
            "first guaranteed income payment",
            "end of first policy month",
            "end of 1st policy month",
            "end of first policy year",
            "end of 1st policy year",
            "guaranteed income",
            "cash bonus",
            "payable on the 13 policy monthiversary",
            "shall be paid till the end of policy term",
        ),
    ),
    _IntentRule(
        "option_count",
        (
            "how many",
            "annuity options can i choose",
        ),
        ("Benefits", "Premiums"),
        ("General", "Policy Terms"),
        (
            "8 annuity options to choose from",
            "2 annuity options to choose from",
            "8 immediate annuity options",
            "2 deferred annuity options",
        ),
    ),
    _IntentRule(
        "early_restriction",
        (
            "early surrender",
            "withdrawal restrictions",
            "surrender or withdrawal restrictions",
            "get money out",
        ),
        ("Surrender",),
        ("Policy Terms", "Premiums"),
        (
            "lock in period",
            "lock-in period",
            "partial withdrawal",
            "partial withdrawals",
            "surrender proceeds",
            "proceeds of the discontinued policy",
            "guaranteed surrender value",
        ),
    ),
    _IntentRule(
        "named_feature",
        (
            "legacy rop",
            "spouse cover",
        ),
        ("Benefits", "Policy Terms"),
        ("General", "Premiums"),
        (
            "legacy rop",
            "transfer of basic sum assured",
            "secondary life insured",
            "spouse cover",
            "available at inception",
            "50% to 100%",
        ),
    ),
    _IntentRule(
        "insta_cashback",
        (
            "insta cashback",
            "cashback feature",
        ),
        ("Benefits", "General"),
        ("Policy Terms", "Premiums"),
        (
            "insta cashback",
            "50% of annualized premium",
            "20% or 30% or 50%",
            "within seven working days",
            "upon policy issuance",
        ),
    ),
    _IntentRule(
        "death_benefit",
        ("sum assured on death", "death benefit", "benefit on death"),
        ("Benefits",),
        ("Policy Terms",),
        ("sum assured on death", "death benefit", "highest of"),
    ),
    _IntentRule(
        "maturity_benefit",
        (
            "maturity benefit",
            "maturity benefits",
            "sum assured on maturity",
            "survival benefit",
            "on survival",
            "paid at maturity",
            "payable at maturity",
            "fund value at maturity",
            "maturity payout",
            "maturity proceeds",
            "policy matures",
            "at maturity",
            "end of the policy term",
        ),
        ("Benefits",),
        ("Policy Terms", "Premiums"),
        _MATURITY_ANCHORS,
    ),
    _IntentRule(
        "surrender",
        ("surrender", "surrender value", "lock in period", "lock-in period"),
        ("Surrender",),
        ("Benefits", "Policy Terms"),
        ("surrender value", "fund value", "policy terminates"),
    ),
    _IntentRule(
        "eligibility",
        ("entry age", "minimum age", "maximum age", "eligibility"),
        ("Eligibility",),
        ("Policy Terms",),
        ("entry age", "minimum age", "maximum age"),
    ),
    _IntentRule(
        "waiting_or_grace",
        ("waiting period", "grace period"),
        ("Policy Terms",),
        ("Eligibility", "Premiums"),
        ("waiting period", "grace period"),
    ),
    _IntentRule(
        "annuity_or_deferment",
        ("deferment period", "deferred annuity", "annuity option"),
        ("Benefits", "Premiums"),
        ("Policy Terms",),
        ("deferment period", "annuity payout", "deferred annuity"),
    ),
    _IntentRule(
        "paid_up_additions",
        ("paid up additions", "paid-up additions", "encash"),
        ("Benefits",),
        ("Policy Terms", "Surrender"),
        ("paid up additions", "paid-up additions", "encash"),
    ),
    _IntentRule(
        "loyalty_additions",
        ("loyalty additions", "guaranteed loyalty additions"),
        ("Benefits", "Premiums"),
        ("Policy Terms",),
        ("guaranteed loyalty additions", "policy year"),
    ),
    _IntentRule(
        "premium",
        ("premium", "premium payment"),
        ("Premiums",),
        ("Policy Terms",),
        ("premium payment",),
    ),
)


def normalize_text(text: str) -> str:
    tokens = [_NUMBER_WORDS.get(token, token) for token in _TOKEN_RE.findall(text.casefold())]
    return " ".join(tokens)


def query_tokens(query: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in normalize_text(query).split()
        if token not in _STOPWORDS and len(token) > 1
    )


def detect_query_intent(query: str) -> QueryIntent:
    """Detect the most specific insurance clause intent in a query."""

    normalized = normalize_text(query)
    matched = [
        rule
        for rule in _INTENT_RULES
        if any(normalize_text(pattern) in normalized for pattern in rule.patterns)
    ]
    if not matched:
        return QueryIntent("general", (), (), (), bool(_SPECIFIC_QUERY.search(query)))
    # ``premium`` is intentionally the final, generic rule. It must never
    # broaden or override a more specific charge, exclusion, benefit, or
    # strategy intent merely because the query also contains "premium".
    if matched[0].name != "premium":
        matched = [rule for rule in matched if rule.name != "premium"]
    shadowed_generic_rules = {
        "bonus_payout": {"paid_up_additions"},
        "early_restriction": {"surrender"},
        "option_count": {"annuity_or_deferment"},
    }
    shadowed = shadowed_generic_rules.get(matched[0].name, set())
    matched = [rule for rule in matched if rule.name not in shadowed]
    # Rules are ordered from more specific to more general. Merge any compatible
    # matches while retaining the first rule's diagnostic name.
    preferred = tuple(
        dict.fromkeys(section for rule in matched for section in rule.preferred_sections)
    )
    compatible = tuple(
        section
        for section in dict.fromkeys(
            section for rule in matched for section in rule.compatible_sections
        )
        if section not in preferred
    )
    anchors = tuple(
        dict.fromkeys(phrase for rule in matched for phrase in rule.anchor_phrases)
    )
    return QueryIntent(
        matched[0].name,
        preferred,
        compatible,
        anchors,
        True,
    )


def section_adjustment(
    section_type: str,
    intent: QueryIntent,
    match_boost: float,
    mismatch_penalty: float,
    text: str = "",
) -> float:
    if not intent.specific or not intent.preferred_sections:
        return 0.0
    if intent.name == "maturity_benefit":
        normalized = normalize_text(text)
        anchor_score = maturity_anchor_score(text)
        if normalized.count("reduced paid up") >= 6:
            # A general maturity query should not be led by a clause devoted
            # exclusively to the reduced-paid-up variation.
            return -2.0 * mismatch_penalty
        if anchor_score >= 0.75 and "maturity benefit" in normalized:
            # A heading plus a payout statement identifies a detailed clause
            # even when recursive chunking inherited an adjacent section label.
            return 0.5 * mismatch_penalty
        if anchor_score >= 0.75 and section_type in intent.compatible_sections:
            return 0.5 * mismatch_penalty
    if section_type in intent.preferred_sections:
        return match_boost
    if section_type in intent.compatible_sections:
        return 0.0
    if intent.name == "maturity_benefit" and maturity_anchor_score(text) >= 0.75:
        # Strong answer text can cross a noisy chunk boundary, but keep a small
        # section penalty so a detailed benefit clause outranks a brochure
        # summary or an adjacent surrender clause.
        return -0.5 * mismatch_penalty
    return -mismatch_penalty


def maturity_anchor_score(text: str) -> float:
    """Return maturity-payout evidence strength on a 0–1 scale.

    A maturity word or heading alone is not evidence. Strong scores require a
    payout component and payment language within the same local clause.
    """

    normalized = normalize_text(text)
    strong_patterns = (
        r"sum assured on maturity.{0,90}"
        r"(?:will be paid|shall be paid|is payable|plus|lump sum)",
        r"fund value.{0,140}(?:inclusive of loyalty additions.{0,40})?"
        r"(?:shall be payable|will be paid|is payable)",
        r"(?:on survival|survival of life insured).{0,180}"
        r"(?:fund value|sum assured on maturity).{0,120}"
        r"(?:shall be payable|will be paid|is payable|plus)",
        r"(?:maturity benefit|at maturity).{0,140}"
        r"(?:fund value|sum assured on maturity).{0,100}"
        r"(?:shall be payable|will be paid|is payable|receive)",
    )
    if any(re.search(pattern, normalized) for pattern in strong_patterns):
        return 1.0
    if re.search(
        r"accumulated guaranteed income.{0,160}"
        r"(?:paid|payable).{0,100}(?:last payout|end of policy term|maturity)",
        normalized,
    ):
        return 0.75
    if any(normalize_text(anchor) in normalized for anchor in _MATURITY_ANCHORS):
        return 0.75
    return 0.0


def intent_anchor_score(text: str, intent: QueryIntent) -> float:
    """Score answer-bearing text anchors for the detected intent."""

    if intent.name == "maturity_benefit":
        return maturity_anchor_score(text)
    if intent.name in _ANSWER_ANCHOR_INTENTS:
        normalized = normalize_text(text)
        matches = sum(
            normalize_text(anchor) in normalized for anchor in intent.anchor_phrases
        )
        return min(1.0, matches / 2.0)
    return 0.0


def answer_evidence_score(window: str, intent: QueryIntent) -> float:
    """Prefer windows that state an answer instead of only naming an option."""

    normalized = normalize_text(window)
    verb_score = min(
        1.0,
        sum(normalize_text(pattern) in normalized for pattern in _ANSWER_VERB_PATTERNS)
        / 2.0,
    )
    if intent.name in _ANSWER_ANCHOR_INTENTS:
        return min(
            1.0,
            0.65 * intent_anchor_score(window, intent) + 0.35 * verb_score,
        )
    return verb_score


def _query_phrases(query: str, intent: QueryIntent) -> tuple[str, ...]:
    normalized = normalize_text(query)
    phrases = [
        normalize_text(pattern)
        for rule in _INTENT_RULES
        for pattern in rule.patterns
        if rule.name == intent.name
        if normalize_text(pattern) in normalized and len(normalize_text(pattern).split()) >= 2
    ]
    phrases.extend(normalize_text(phrase) for phrase in intent.anchor_phrases)
    return tuple(dict.fromkeys(phrase for phrase in phrases if phrase))


def exact_match_score(query: str, text: str, intent: QueryIntent) -> float:
    """Score distinctive phrases, query terms, and numbers on a 0–1 scale."""

    normalized_text = normalize_text(text)
    terms = set(query_tokens(query))
    term_coverage = (
        sum(term in normalized_text.split() for term in terms) / len(terms) if terms else 0.0
    )
    phrases = _query_phrases(query, intent)
    phrase_coverage = (
        sum(phrase in normalized_text for phrase in phrases) / len(phrases)
        if phrases
        else 0.0
    )
    query_numbers = {token for token in normalize_text(query).split() if token.isdigit()}
    text_numbers = {token for token in normalized_text.split() if token.isdigit()}
    number_coverage = (
        len(query_numbers & text_numbers) / len(query_numbers) if query_numbers else 0.0
    )
    calculation_anchor = (
        1.0
        if intent.name == "death_benefit"
        and _SPECIFIC_QUERY.search(query)
        and "highest of" in normalized_text
        else 0.0
    )
    return min(
        1.0,
        0.25 * term_coverage
        + 0.35 * phrase_coverage
        + 0.20 * number_coverage
        + 0.20 * calculation_anchor,
    )


def lexical_similarity(a: str, b: str) -> float:
    """Cheap, deterministic near-duplicate signal between two chunk texts.

    Overlapping brochure chunks (recursive-chunking windows, repeated clauses
    across sections) share most of their normalized token sequence. Comparing
    that sequence directly avoids an embedding forward pass per candidate.
    """

    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def _split_long_unit(unit: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = unit.split()
    if len(tokens) <= max_tokens:
        return [unit]
    step = max(1, max_tokens - overlap_tokens)
    return [
        " ".join(tokens[start : start + max_tokens])
        for start in range(0, len(tokens), step)
        if tokens[start : start + max_tokens]
    ]


def sentence_windows(
    text: str,
    max_tokens: int = 110,
    overlap_tokens: int = 25,
) -> list[str]:
    """Create short sentence/paragraph windows without changing the parent chunk."""

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph.strip()
    ]
    units: list[str] = []
    for paragraph in paragraphs:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
            if sentence.strip()
        ]
        for sentence in sentences or [paragraph]:
            units.extend(_split_long_unit(sentence, max_tokens, overlap_tokens))

    windows: list[str] = []
    for start in range(len(units)):
        combined: list[str] = []
        token_count = 0
        for unit in units[start:]:
            unit_count = len(unit.split())
            if combined and token_count + unit_count > max_tokens:
                break
            combined.append(unit)
            token_count += unit_count
            if token_count >= max_tokens:
                break
        if combined:
            windows.append(" ".join(combined))
    if not windows:
        windows = _split_long_unit(re.sub(r"\s+", " ", text).strip(), max_tokens, overlap_tokens)
    return list(dict.fromkeys(window for window in windows if window))


def lexical_window_score(
    query: str,
    window: str,
    intent: QueryIntent,
) -> float:
    terms = set(query_tokens(query))
    window_tokens = set(normalize_text(window).split())
    term_coverage = len(terms & window_tokens) / len(terms) if terms else 0.0
    phrase_score = exact_match_score(query, window, intent)
    evidence_score = answer_evidence_score(window, intent)
    if intent.name in _ANSWER_ANCHOR_INTENTS:
        return min(
            1.0,
            0.45 * term_coverage + 0.20 * phrase_score + 0.35 * evidence_score,
        )
    return min(1.0, 0.55 * term_coverage + 0.30 * phrase_score + 0.15 * evidence_score)


def select_focused_windows(
    query: str,
    texts: Sequence[str],
    intent: QueryIntent,
    embedder: Embedder | None,
    lexical_candidates_per_chunk: int = 3,
    semantic_weight: float = 0.65,
) -> list[FocusedWindow]:
    """Select one query-focused window per chunk using lexical and embedding scores."""

    if lexical_candidates_per_chunk < 1:
        raise ValueError("lexical_candidates_per_chunk must be positive")
    shortlisted: list[list[tuple[str, float]]] = []
    flat_windows: list[str] = []
    for text in texts:
        ranked = sorted(
            (
                (
                    window,
                    lexical_window_score(query, window, intent),
                )
                for window in sentence_windows(text)
            ),
            key=lambda item: (-item[1], len(item[0])),
        )[:lexical_candidates_per_chunk]
        shortlisted.append(ranked)
        flat_windows.extend(window for window, _ in ranked)

    semantic_scores: Iterable[float | None]
    if embedder is not None and flat_windows:
        query_vector = np.asarray(embedder.embed_query(query), dtype=float)
        window_vectors = np.asarray(embedder.embed_documents(flat_windows), dtype=float)
        semantic_scores = np.clip(window_vectors @ query_vector, -1.0, 1.0).tolist()
    else:
        semantic_scores = [None] * len(flat_windows)

    score_iterator = iter(semantic_scores)
    selected: list[FocusedWindow] = []
    for ranked in shortlisted:
        options: list[FocusedWindow] = []
        for window, lexical_score in ranked:
            embedding_score = next(score_iterator)
            combined = (
                semantic_weight * max(0.0, embedding_score)
                + (1.0 - semantic_weight) * lexical_score
                if embedding_score is not None
                else lexical_score
            )
            options.append(
                FocusedWindow(window, combined, lexical_score, embedding_score)
            )
        selected.append(max(options, key=lambda item: item.score))
    return selected
