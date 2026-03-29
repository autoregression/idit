import copy
import typing

import datasets
import pydantic_settings as pyds
import torch
import torchvision
import tqdm

from src.idit.model import IDiT, IDiTConfig


class IDiTTrainerConfig(pyds.BaseSettings):
    seed: int = 0

    # Data.

    dataset_path: str = "mnist"
    split: str = "train"
    column: str = "image"

    # Model.

    input_height: int = 16
    input_width: int = 16
    input_dimension: int = 1
    hidden_dimension: int = 64
    head_dimension: int = 16
    condition_dimension: int = 64
    frequency_dimension: int = 256
    layers: int = 2
    iterations: int = 4

    # Optimizer.

    steps: int = 10_000
    batch_size: int = 4
    gradient_accumulation: int = 1
    learning_rate: float = 1e-3
    warmup: int = 100
    ema: int = 1000
    ema_beta: float = 0.999
    ema_exponent: float = 10.0

    # Tracker.

    checkpoint_path: str = "checkpoint"


class IDiTTrainer(typing.NamedTuple):
    config: IDiTTrainerConfig

    def train(self) -> None:
        torch.manual_seed(self.config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float32

        # Data.

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.config.input_height, self.config.input_width)),
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
                input_height=self.config.input_height,
                input_width=self.config.input_width,
                input_dimension=self.config.input_dimension,
                hidden_dimension=self.config.hidden_dimension,
                head_dimension=self.config.head_dimension,
                condition_dimension=self.config.condition_dimension,
                frequency_dimension=self.config.frequency_dimension,
                layers=self.config.layers,
                iterations=self.config.iterations,
            )
        ).to(device, dtype)

        ema_model = copy.deepcopy(model)

        # Optimizer.

        def warmup_constant(step: int) -> float:
            if step < self.config.warmup:
                return step / self.config.warmup

            return 1.0

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_constant)

        for step in tqdm.trange(self.config.steps, colour="green", desc="Training"):
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

            if (step + 1) % max(self.config.steps // self.config.ema, 1) == 0:
                with torch.no_grad():
                    for parameter, ema_parameter in zip(model.parameters(), ema_model.parameters()):
                        progress = (step + 1) / self.config.steps
                        beta = self.config.ema_beta - (1 - progress) ** self.config.ema_exponent
                        ema_parameter.data = beta * ema_parameter.data + (1 - beta) * parameter.data

        ema_model.save_checkpoint(self.config.checkpoint_path)
