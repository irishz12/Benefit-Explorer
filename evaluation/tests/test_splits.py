import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "evaluation" / "golden" / "golden_questions.json"
SPLITS_PATH = ROOT / "evaluation" / "golden" / "splits.json"

EXPECTED_DEV_IDS = (
    "Q001",
    "Q004",
    "Q005",
    "Q006",
    "Q007",
    "Q009",
    "Q011",
    "Q013",
    "Q015",
    "Q017",
    "Q021",
    "Q023",
    "Q025",
    "Q027",
    "Q028",
    "Q030",
    "Q032",
    "Q033",
    "Q035",
    "Q036",
)
EXPECTED_HOLDOUT_IDS = (
    "Q002",
    "Q003",
    "Q008",
    "Q012",
    "Q016",
    "Q022",
    "Q026",
    "Q029",
    "Q031",
    "Q034",
)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dev_and_holdout_splits_are_disjoint_complete_and_stable() -> None:
    split = _load_json(SPLITS_PATH)
    golden = _load_json(GOLDEN_PATH)
    assert isinstance(split, dict)
    assert isinstance(golden, list)

    dev_ids = tuple(split["dev_question_ids"])
    holdout_ids = tuple(split["holdout_question_ids"])
    golden_ids = {row["question_id"] for row in golden}

    assert dev_ids == EXPECTED_DEV_IDS
    assert holdout_ids == EXPECTED_HOLDOUT_IDS
    assert len(dev_ids) == 20
    assert len(holdout_ids) == 10
    assert set(dev_ids).isdisjoint(holdout_ids)
    assert set(dev_ids) | set(holdout_ids) == golden_ids
    assert set(split["question_types"]) == golden_ids


def test_both_splits_cover_required_question_types_and_all_products() -> None:
    split = _load_json(SPLITS_PATH)
    golden = _load_json(GOLDEN_PATH)
    assert isinstance(split, dict)
    assert isinstance(golden, list)

    products = {row["question_id"]: row.get("product") for row in golden}
    types = split["question_types"]
    required_types = {"factual", "comparison", "exclusion_or_limitation"}
    named_products = {product for product in products.values() if product is not None}

    for key in ("dev_question_ids", "holdout_question_ids"):
        ids = set(split[key])
        assert required_types <= {types[question_id] for question_id in ids}
        assert named_products <= {products[question_id] for question_id in ids}
