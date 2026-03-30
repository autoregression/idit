import os
import typing

import pydantic_settings as pyds
import torch
import tqdm

from src.idit.model import IDiT
from src.idit.shared import dtype, acc_device as device, load_timestamp_path, save_image_stack


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

        checkpoint_folder = load_timestamp_path(self.config.checkpoint_path)
        model = IDiT.from_checkpoint(checkpoint_folder).to(device, dtype)

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

        save_image_stack(samples=noisy, timestamp=checkpoint_folder.stem)
