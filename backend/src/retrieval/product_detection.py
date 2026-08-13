"""Product-name detection for product-aware insurance retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


def _normalize(value: str) -> str:
    value = value.casefold().replace("maximizer", "maximiser")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


@dataclass(frozen=True, slots=True)
class ProductMatch:
    canonical_name: str
    matched_alias: str
    position: int


class ProductDetector:
    """Resolve common aliases to canonical product metadata values."""

    _KNOWN_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("assured pension", ("kotak assured pension", "assured pension")),
        (
            "fortune maximiser",
            ("kotak fortune maximiser", "fortune maximiser", "fortune maximizer"),
        ),
        (
            "gen2gen",
            ("kotak gen2gen protect", "gen2gen protect", "gen2gen", "gen 2 gen"),
        ),
        ("tulip", ("kotak tulip", "tulip", "t u l i p")),
        ("edge", ("kotak edge", "edge")),
        # "gain" is common English. Requiring the brand-qualified name avoids
        # routing ordinary phrases such as "gain better value" to this product.
        ("gain", ("kotak gain",)),
    )

    def __init__(self, product_names: Iterable[str]) -> None:
        self.product_names = tuple(dict.fromkeys(product_names))
        aliases: list[tuple[str, str]] = []
        for product_name in self.product_names:
            normalized_product = _normalize(product_name)
            dynamic_aliases = {normalized_product}
            if normalized_product.startswith("kotak ") and normalized_product != "kotak gain":
                dynamic_aliases.add(normalized_product.removeprefix("kotak "))
            for selector, known_aliases in self._KNOWN_ALIASES:
                if selector in normalized_product:
                    dynamic_aliases.update(_normalize(alias) for alias in known_aliases)
            aliases.extend((alias, product_name) for alias in dynamic_aliases if alias)
        self._aliases = tuple(
            sorted(set(aliases), key=lambda item: (-len(item[0]), item[1].casefold()))
        )

    def detect(self, query: str) -> tuple[str, ...]:
        """Return canonical products in the order they are mentioned."""

        normalized_query = _normalize(query)
        matches: list[ProductMatch] = []
        seen: set[str] = set()
        for alias, product_name in self._aliases:
            match = re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                normalized_query,
            )
            if match is None or product_name in seen:
                continue
            seen.add(product_name)
            matches.append(ProductMatch(product_name, alias, match.start()))
        matches.sort(key=lambda match: (match.position, -len(match.matched_alias)))
        return tuple(match.canonical_name for match in matches)
