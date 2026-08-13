from src.retrieval.product_detection import ProductDetector


def test_gain_requires_brand_qualified_name() -> None:
    detector = ProductDetector(("Kotak GAIN", "Kotak EDGE"))
    assert detector.detect("How can I gain better surrender value?") == ()
    assert detector.detect("What is the gain from early surrender?") == ()
    assert detector.detect("Tell me about Kotak GAIN surrender value") == ("Kotak GAIN",)


def test_unambiguous_short_aliases_still_work() -> None:
    detector = ProductDetector(("Kotak GAIN", "Kotak EDGE"))
    assert detector.detect("Compare EDGE with Kotak GAIN") == (
        "Kotak EDGE",
        "Kotak GAIN",
    )
