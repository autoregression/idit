# test/test_timestamp.py
import pytest
from pathlib import Path
from idit.shared import list_timestamp_paths


def test_list_timestamp_paths_empty_folder(tmp_path, monkeypatch):
    # Create a fake ROOT_FOLDER in temp location
    fake_root = tmp_path / "root"
    fake_root.mkdir()

    # Monkeypatch ROOT_FOLDER to point to our temp directory
    monkeypatch.setattr("idit.shared.ROOT_FOLDER", fake_root)

    result = list_timestamp_paths("checkpoint")
    assert result == []


def test_list_timestamp_paths_valid_timestamps(tmp_path, monkeypatch):
    fake_root = tmp_path / "root"
    fake_root.mkdir()

    # Create some timestamped folders
    (fake_root / "checkpoint" / "20241025_143218").mkdir(parents=True)
    (fake_root / "checkpoint" / "20240920_100000").mkdir(parents=True)
    (fake_root / "checkpoint" / "invalid_name").mkdir(parents=True)  # Should be filtered

    monkeypatch.setattr("idit.shared.ROOT_FOLDER", fake_root)

    result = list_timestamp_paths("checkpoint")

    # Should be sorted descending
    assert result == ["20241025_143218", "20240920_100000"]


if __name__ == "main":
    pytest.main["-v"]
