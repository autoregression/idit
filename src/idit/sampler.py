import os
import typing

import pydantic_settings as pyds
import torch
import torchvision
import tqdm

from src.idit.model import IDiT
from src.idit.shared import TIMESTAMP_PATTERN, ROOT_FOLDER, list_timestamp_paths, dtype, acc_device as device


class IDiTSamplerConfig(pyds.BaseSettings):
    seed: int = 0
    resolution: int = 28
    steps: int = 20
    batch_size: int = 16
    samples_path: str = "samples"
    checkpoint_path: str = "checkpoint"


class IDiTSampler(typing.NamedTuple):
    config: IDiTSamplerConfig

    @torch.inference_mode()
    def sample(self) -> None:
        torch.manual_seed(self.config.seed)

        if TIMESTAMP_PATTERN.fullmatch(self.config.checkpoint_path):
            model_folder = ROOT_FOLDER / "checkpoint" / self.config.checkpoint_path
        else:
            if paths := list_timestamp_paths():
                model_folder = ROOT_FOLDER / "checkpoint" / paths[0]
            else:
                raise FileNotFoundError(f"No checkpoint folder found in {ROOT_FOLDER / 'checkpoint'}")

        model = IDiT.from_checkpoint(self.config.checkpoint_path).to(device, dtype)

        noisy = torch.randn(
            self.config.batch_size,
            model.config.input_dimension,
            self.config.resolution,
            self.config.resolution,
            device=device,
            dtype=dtype,
        )

        for step in tqdm.trange(self.config.steps, colour="blue", desc="Sampling"):
            time = torch.full((noisy.size(0),), 1 - step / self.config.steps, device=noisy.device, dtype=noisy.dtype)
            prediction = model.predict(noisy, time=time)
            noisy = noisy - prediction / self.config.steps  # https://arxiv.org/abs/2209.03003

        os.makedirs(self.config.samples_path, exist_ok=True)

        for i, sample in enumerate(noisy):
            image = torchvision.transforms.ToPILImage()(sample.clip(0, 1))
            image = image.resize((256, 256), 0)
            image_path = ROOT_FOLDER / self.config.samples_path / model_folder.name
            image_path.mkdir(exist_ok=True)
            image.save(f"{image_path}/sample_{i:03d}.png")
