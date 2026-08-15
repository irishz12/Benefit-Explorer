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


def test_legacy_rop_routes_to_gen2gen_protect() -> None:
    """Legacy ROP is a distinctive named feature unique to Kotak Gen2Gen Protect."""

    detector = ProductDetector(("Kotak Gen2Gen Protect", "Kotak EDGE"))
    assert detector.detect(
        "How does the Legacy ROP option protect my child after my policy term ends?"
    ) == ("Kotak Gen2Gen Protect",)
    assert detector.detect("What age conditions apply for the Legacy ROP option?") == (
        "Kotak Gen2Gen Protect",
    )


def test_premium_saver_routes_to_gain_without_widening_bare_gain() -> None:
    """"Premium Saver" is exclusive to Kotak GAIN, so it is safe to detect on its
    own — unlike the bare word "gain", which stays brand-qualified."""

    detector = ProductDetector(("Kotak GAIN", "Kotak EDGE"))
    assert detector.detect(
        "How can the Guaranteed Loyalty Additions under GAIN's Premium Saver "
        "option help pay my premiums?"
    ) == ("Kotak GAIN",)
    assert detector.detect("Tell me about the Premium Saver option.") == ("Kotak GAIN",)
    # Generic English "gain" must still require the brand-qualified name.
    assert detector.detect("How can I gain better surrender value?") == ()
    assert detector.detect("What is the gain from early surrender?") == ()
