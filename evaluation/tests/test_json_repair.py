from evaluation.src.ragas_metrics import repair_json_object


def test_valid_json_is_returned_unchanged() -> None:
    payload = '{"statements": ["a", "b"]}'
    assert repair_json_object(payload) == payload


def test_duplicated_opening_brace_is_repaired() -> None:
    assert repair_json_object('{\n{"statements": ["a"]}') == '{"statements": ["a"]}'


def test_partial_key_prefill_is_repaired() -> None:
    assert repair_json_object('{\n  "{"statements": ["a"]}') == '{"statements": ["a"]}'


def test_unrecoverable_content_is_left_for_the_parser_to_reject() -> None:
    broken = '{"statements": ['
    assert repair_json_object(broken) == broken
