from pathlib import Path

from src.main import BACKEND_DIR, DEFAULT_CHROMA, resolve_chroma_dir


def test_unset_value_falls_back_to_the_backend_default() -> None:
    assert resolve_chroma_dir(None) == DEFAULT_CHROMA
    assert resolve_chroma_dir("") == DEFAULT_CHROMA


def test_relative_value_resolves_against_backend_not_the_working_directory() -> None:
    assert resolve_chroma_dir("data/chroma_db") == BACKEND_DIR / "data" / "chroma_db"


def test_absolute_value_is_used_as_given() -> None:
    assert resolve_chroma_dir("/srv/index") == Path("/srv/index")
