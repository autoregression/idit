import typing

import datasets
import pydantic_settings as pyds
import torch
import torchvision
import tqdm

from src.idit.model import IDiT, IDiTConfig
from src.idit.shared import dtype, acc_device as device


class IDiTTrainerConfig(pyds.BaseSettings):
    seed: int = 0

    # Data.

    dataset_path: str = "mnist"
    split: str = "train"
    column: str = "image"
    resolution: int = 28

    # Model.

    input_dimension: int = 1
    hidden_dimension: int = 64
    head_dimension: int = 16
    condition_dimension: int = 64
    frequency_dimension: int = 256
    layers: int = 1
    iterations: int = 8
    patch_size: int = 2

    # Optimizer.

    steps: int = 20_000
    batch_size: int = 4
    gradient_accumulation: int = 1
    learning_rate: float = 1e-3
    warmup: int = 100
    cooldown: int = 500

    # Checkpointer.

    checkpoint_path: str = "checkpoint"


class IDiTTrainer(typing.NamedTuple):
    config: IDiTTrainerConfig

    def train(self) -> None:
        torch.manual_seed(self.config.seed)

        # Data.

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.config.resolution, self.config.resolution)),
                torchvision.transforms.ToTensor(),
            ]
        )

        dataset = datasets.load_dataset(self.config.dataset_path, split=self.config.split)
        dataset.set_transform(lambda examples: {self.config.column: [transform(x) for x in examples[self.config.column]]})
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)  #  type: ignore

        def create_batches():
            while True:
                for batch in data_loader:
                    yield batch[self.config.column]

        batches = create_batches()

        # Model.

        model = IDiT(
            config=IDiTConfig(
                input_dimension=self.config.input_dimension,
                hidden_dimension=self.config.hidden_dimension,
                head_dimension=self.config.head_dimension,
                condition_dimension=self.config.condition_dimension,
                frequency_dimension=self.config.frequency_dimension,
                layers=self.config.layers,
                iterations=self.config.iterations,
                patch_size=self.config.patch_size,
            )
        ).to(device, dtype)

        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"Training {parameters / 1e6:.2f}M parameters on {device}")

        # Optimizer.

        def warmup_stable_decay(step: int) -> float:
            if step < self.config.warmup:
                return step / self.config.warmup

            if step < self.config.steps - self.config.cooldown:
                return 1.0

            return max(0, 1 - (step - (self.config.steps - self.config.cooldown)) / self.config.cooldown)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_stable_decay)

        for _ in tqdm.trange(self.config.steps, colour="green", desc="Training"):
            total_loss = 0.0
            optimizer.zero_grad()

            for _ in range(self.config.gradient_accumulation):
                data = next(batches).to(device, dtype)
                loss = model(data) / self.config.gradient_accumulation
                loss.backward()
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.save_checkpoint(self.config.checkpoint_path)
