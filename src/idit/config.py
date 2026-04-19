"""Consolidated configuration presets across all datasets."""

import typing
import pydantic as pyd
import pydantic_settings as pyds


DATASET_CONFIGS: dict[str, dict[str, typing.Any]] = {
    "bitmind/ffhq-256": {
        "resolution": 256,
        "steps": 100_000,
        "patch_size": 16,
        "input_dimension": 3,
    },
    "mnist": {
        "resolution": 28,
        "steps": 20_000,
        "patch_size": 2,
        "input_dimension": 1,
    },
}


class IDiTPresets(pyds.BaseSettings):
    dataset_path: str = "bitmind/ffhq-256"

    # shared

    seed: int = 0
    resolution: int = 256
    steps: int = 100_000
    batch_size: int = 4
    checkpoint_path: str = "checkpoint"

    # training

    split: str = "train"
    column: str = "image"
    input_dimension: int = 3
    hidden_dimension: int = 64
    head_dimension: int = 16
    condition_dimension: int = 64
    frequency_dimension: int = 256
    layers: int = 1
    iterations: int = 8
    patch_size: int = 16
    gradient_accumulation: int = 1
    learning_rate: float = 1e-3
    warmup: int = 100
    cooldown: int = 500

    # inference_only
    inference_steps: int = 20
    inference_batch_size: int = 16
    samples_path: str = "samples"

    def with_dataset(self) -> "IDiTPresets":
        dataset_overrides = DATASET_CONFIGS.get(self.dataset_path, {})

        user_values = {}
        for field_name, field_info in IDiTPresets.model_fields.items():
            current_value = getattr(self, field_name)
            default_value = field_info.default
            if hasattr(default_value, "__dataclass_fields__"):
                continue
            if current_value != default_value:
                user_values[field_name] = current_value

        return IDiTPresets(**{**dataset_overrides, **user_values})


class IDiTConfig(pyd.BaseModel):
    input_dimension: int
    hidden_dimension: int
    head_dimension: int
    condition_dimension: int
    frequency_dimension: int
    layers: int
    iterations: int
    patch_size: int
    t_eps: float = 5e-2
    jit_type: bool = True


class IDiTSamplerConfig(pyd.BaseModel):
    seed: int
    resolution: int
    inference_steps: int
    inference_batch_size: int
    samples_path: str
    checkpoint_path: str

    @classmethod
    def from_presets(cls, presets: IDiTPresets) -> "IDiTSamplerConfig":
        return cls(
            seed=presets.seed,
            resolution=presets.resolution,
            inference_steps=presets.inference_steps,
            inference_batch_size=presets.inference_batch_size,
            samples_path=presets.samples_path,
            checkpoint_path=presets.checkpoint_path,
        )


class IDiTTrainerConfig(pyd.BaseModel):
    seed: int
    dataset_path: str
    split: str
    column: str
    resolution: int

    input_dimension: int
    hidden_dimension: int
    head_dimension: int
    condition_dimension: int
    frequency_dimension: int
    layers: int
    iterations: int
    patch_size: int

    steps: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    warmup: int
    cooldown: int

    checkpoint_path: str

    @classmethod
    def from_presets(cls, presets: IDiTPresets) -> "IDiTTrainerConfig":
        return cls(
            seed=presets.seed,
            dataset_path=presets.dataset_path,
            split=presets.split,
            column=presets.column,
            resolution=presets.resolution,
            input_dimension=presets.input_dimension,
            hidden_dimension=presets.hidden_dimension,
            head_dimension=presets.head_dimension,
            condition_dimension=presets.condition_dimension,
            frequency_dimension=presets.frequency_dimension,
            layers=presets.layers,
            iterations=presets.iterations,
            patch_size=presets.patch_size,
            steps=presets.steps,
            batch_size=presets.batch_size,
            gradient_accumulation=presets.gradient_accumulation,
            learning_rate=presets.learning_rate,
            warmup=presets.warmup,
            cooldown=presets.cooldown,
            checkpoint_path=presets.checkpoint_path,
        )
