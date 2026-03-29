from datetime import datetime
import re
import pytest
from unittest.mock import patch


class TestNewTimestampPath:
    """Test suite for new_timestamp_path function."""

    def test_creates_folder_at_expected_location(self, tmp_path):
        """Verify folder is created in the correct location."""
        from idit.shared import new_timestamp_path

        with patch("idit.shared.datetime") as mock_datetime:
            mock_time = datetime(2024, 10, 25, 14, 32, 18)
            mock_datetime.now.return_value = mock_time
            path_name = str(tmp_path / "models")
            result = new_timestamp_path(path_name)

        assert result.exists()
        assert result.is_dir()

    def test_returns_path_with_timestamp_format(self, tmp_path):
        """Verify returned path contains timestamp in expected format."""
        from idit.shared import new_timestamp_path

        with patch("idit.shared.datetime") as mock_datetime:
            mock_time = datetime(2024, 10, 25, 14, 32, 18)
            mock_datetime.now.return_value = mock_time
            path_name = str(tmp_path / "output")
            result = new_timestamp_path(path_name)

        pattern = r"output[/\\]20241025_143218$"
        assert re.search(pattern, str(result))

    def test_creates_nested_directories(self, tmp_path):
        """Verify parent directories are created when they don't exist."""
        from idit.shared import new_timestamp_path

        with patch("idit.shared.datetime") as mock_datetime:
            mock_time = datetime(2023, 1, 1, 0, 0, 0)
            mock_datetime.now.return_value = mock_time
            deep_path = str(tmp_path / "deep" / "nested" / "folder")
            result = new_timestamp_path(deep_path)

        assert (tmp_path / "deep" / "nested").exists()


if __name__ == "main":
    pytest.main["-v"]
