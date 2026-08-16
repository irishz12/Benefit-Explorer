from __future__ import annotations

from src.generation.ranking_signals import detect_query_intent, lexical_similarity


def test_get_money_out_maps_to_early_restriction_intent() -> None:
    """Q008 asks to "get money out" during the lock-in. That must resolve to the
    surrender/withdrawal-restriction intent so surrender-and-lock-in chunks are
    preferred over a generic brochure page — the root cause of the wrong answer."""

    intent = detect_query_intent(
        "Can I get money out of Kotak TULIP during the first five policy years?"
    )
    assert intent.name == "early_restriction"
    assert "Surrender" in intent.preferred_sections


def test_lexical_similarity_scores_near_identical_text_high() -> None:
    a = (
        "The maturity benefit shall be payable at the end of the policy term "
        "provided all due premiums have been paid and the policy is in force."
    )
    b = (
        "The maturity benefit shall be payable at the end of the policy term "
        "provided all due premiums have been paid, and the policy remains in force."
    )
    assert lexical_similarity(a, b) > 0.88


def test_lexical_similarity_scores_distinct_text_low() -> None:
    a = "The maturity benefit shall be payable at the end of the policy term."
    b = "Suicide within 12 months of the risk commencement date restricts the death benefit."
    assert lexical_similarity(a, b) < 0.5


def test_lexical_similarity_is_bounded() -> None:
    a = "Guaranteed income shall commence after the deferment period ends."
    b = "The lock-in period restricts partial withdrawals for five policy years."
    assert 0.0 <= lexical_similarity(a, b) <= 1.0
    assert 0.0 <= lexical_similarity(b, a) <= 1.0
