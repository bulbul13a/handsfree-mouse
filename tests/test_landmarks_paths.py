import pytest

from handsfree_mouse.landmarks import MODEL_NAME, resolve_model_path


def test_resolve_explicit_path(tmp_path, monkeypatch):
    monkeypatch.delenv("HANDSFREE_MODEL", raising=False)
    model = tmp_path / MODEL_NAME
    model.write_bytes(b"x")
    assert resolve_model_path(str(model)) == model


def test_resolve_from_env(tmp_path, monkeypatch):
    model = tmp_path / "custom.dat"
    model.write_bytes(b"x")
    monkeypatch.setenv("HANDSFREE_MODEL", str(model))
    assert resolve_model_path("does/not/exist.dat") == model


def test_missing_model_gives_helpful_error(tmp_path, monkeypatch):
    monkeypatch.delenv("HANDSFREE_MODEL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="download_model"):
        resolve_model_path(str(tmp_path / "missing.dat"))
