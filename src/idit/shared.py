import re
from datetime import datetime
from pathlib import Path
from typing import Pattern

import torch

acc_device = torch.accelerator.current_accelerator()
dtype = torch.float32
torch.accelerator.current_accelerator()
ROOT_FOLDER = Path(__file__).resolve().parent.parent.parent
TIMESTAMP_PATTERN: Pattern[str] = re.compile(r"\d{8}_\d{6}")


def list_timestamp_paths(folder_name: str = "checkpoint") -> list[str]:
    """Load all timestamp paths from a specific subfolder.\n
    :param folder_name: Name of the folder. (eg: checkpoint, samples)
    :return: Path object to the timestamped folder location.
    """
    folder_path = ROOT_FOLDER / folder_name
    if not folder_path.exists():
        return []
    entries = [entry.name for entry in folder_path.iterdir() if entry.is_dir() and bool(TIMESTAMP_PATTERN.fullmatch(entry.name))]
    entries.sort(reverse=True)
    return entries


def new_timestamp_path(folder_name: str = "checkpoint") -> str:
    """Generate a filesystem-safe path with embedded timestamp.\n
    :param folder_name: Name of the folder (not including components).
    :return: Path object to the timestamped folder location.
    :raises OSError: If directory creation fails due to permissions.

    Example:
        >>> path = new_timestamp_path("models")
        >>> print(path)
        models/20241025_143218
    """
    get_time = lambda: datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa
    abs_datestamp_path: Path = ROOT_FOLDER / folder_name / get_time()

    abs_datestamp_path.mkdir(parents=True, exist_ok=True)
    generated_path = abs_datestamp_path
    return generated_path
