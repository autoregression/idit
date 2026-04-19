import os
import typing

import torch
import tqdm

from idit.config import IDiTSamplerConfig
from idit.model import IDiT
from idit.shared import acc_device as device, dtype, load_timestamp_path, save_image_stack


class IDiTSampler(typing.NamedTuple):
    config: IDiTSamplerConfig

    @torch.inference_mode()
    def sample(self) -> None:
        torch.manual_seed(self.config.seed)

        checkpoint_folder = load_timestamp_path(self.config.checkpoint_path)
        model = IDiT.from_checkpoint(checkpoint_folder).to(device, dtype)

        noisy = torch.randn(
            self.config.inference_batch_size,
            model.config.input_dimension,
            self.config.resolution,
            self.config.resolution,
            device=device,
            dtype=dtype,
        )

        for step in tqdm.trange(self.config.inference_steps, colour="blue", desc="Sampling"):
            if model.config.jit_type:
                time = torch.full((noisy.size(0),), step / self.config.inference_steps, device=noisy.device, dtype=noisy.dtype)
            else:
                time = torch.full((noisy.size(0),), 1 - step / self.config.inference_steps, device=noisy.device, dtype=noisy.dtype)

            prediction = model.predict(noisy, time=time)

            if model.config.jit_type:
                prediction = (prediction - noisy) / (1 - time.view(-1, 1, 1, 1)).clamp_min(model.config.t_eps)
                noisy = noisy + prediction / self.config.inference_steps  # https://arxiv.org/abs/2511.13720
            else:
                noisy = noisy - prediction / self.config.inference_steps  # https://arxiv.org/abs/2209.03003

        os.makedirs(self.config.samples_path, exist_ok=True)

        save_image_stack(samples=noisy, timestamp=checkpoint_folder.stem)
