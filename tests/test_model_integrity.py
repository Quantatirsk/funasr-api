# -*- coding: utf-8 -*-
"""
Model integrity checker tests.
"""

from pathlib import Path

from app.utils.model_loader import ModelIntegritySpec, _check_model_integrity_spec


def test_model_integrity_rejects_missing_required_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "configuration.json").write_text("{}")

    result = _check_model_integrity_spec(
        ModelIntegritySpec(
            description="broken-model",
            path=model_dir,
            required_patterns=("configuration.json", "model.pt"),
            min_total_size_bytes=1,
        )
    )

    assert result["ok"] is False
    assert result["reason"] == "required_files_missing"
    assert result["missing_patterns"] == ["model.pt"]


def test_model_integrity_accepts_complete_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "configuration.json").write_text("{}")
    (model_dir / "model.pt").write_bytes(b"x" * 2048)

    result = _check_model_integrity_spec(
        ModelIntegritySpec(
            description="complete-model",
            path=model_dir,
            required_patterns=("configuration.json", "model.pt"),
            min_total_size_bytes=1024,
        )
    )

    assert result["ok"] is True
    assert result["reason"] == "ok"
