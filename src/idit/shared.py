import re
from datetime import datetime
from pathlib import Path
import torch
import torchvision


acc_device = torch.accelerator.current_accelerator()
dtype = torch.float32

ROOT_FOLDER = Path(__file__).resolve().parent.parent.parent
TIMESTAMP_PATTERN = re.compile(r"\d{8}_\d{6}")


def list_timestamp_paths(folder_name: str = "checkpoint") -> list[Path]:
    """Load all timestamp paths from a specific subfolder.\n
    :param folder_name: Name of the folder. (eg: checkpoint, samples)
    :return: Sorted list of path obj location, newest first.
    """
    folder_path = ROOT_FOLDER / folder_name
    if not folder_path.exists():
        return []
    entries = [entry.name for entry in folder_path.iterdir() if entry.is_dir() and bool(TIMESTAMP_PATTERN.fullmatch(entry.name))]
    entries.sort(reverse=True)
    return entries


def new_timestamp_path(folder_name: str = "checkpoint") -> Path:
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


def load_timestamp_path(timestamp: str | None = None, folder_name: str = "checkpoint"):
    """Load the most recent checkpoint or sample path or a specific timestamped folder.\n
    :param timestamp: Timestamp string (YYYYMMDD_HHMMSS) or
    :param folder_name: Folder for the load operation.
    :return: Path to the required folder.
    :raises FileNotFoundError: If no checkpoint folders exist and none specified.
    """

    if timestamp is not None and TIMESTAMP_PATTERN.fullmatch(timestamp):
        model_folder = ROOT_FOLDER / folder_name / timestamp
    else:
        if paths := list_timestamp_paths():
            model_folder = ROOT_FOLDER / folder_name / paths[0]
        else:
            raise FileNotFoundError(f"No checkpoint folder found in {ROOT_FOLDER / timestamp}")
    return model_folder


def save_image_stack(samples: torch.Tensor, timestamp: str, folder_name="samples"):
    """Save tensor bundle as PNG image files.\n
    :param sample: Tensor stack with values in range [0, 1].
    :param timestamp: Folder timestamp to save under.
    :param folder_name: Parent folder name (default: samples).
    """

    for index, sample in enumerate(samples):
        image = torchvision.transforms.ToPILImage()(sample.clip(0, 1))
        image = image.resize((256, 256), 0)
        image_path = ROOT_FOLDER / folder_name / timestamp
        image_path.mkdir(exist_ok=True)
        image.save(f"{image_path}/sample_{index:03d}.png")
