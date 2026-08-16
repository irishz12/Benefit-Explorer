from __future__ import annotations

from src.main import _optional_device


def test_optional_device_returns_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    assert _optional_device("EMBEDDING_DEVICE") is None


def test_optional_device_returns_none_when_blank(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_DEVICE", "   ")
    assert _optional_device("RERANKER_DEVICE") is None


def test_optional_device_returns_stripped_value_when_set(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_DEVICE", " cpu ")
    assert _optional_device("EMBEDDING_DEVICE") == "cpu"
