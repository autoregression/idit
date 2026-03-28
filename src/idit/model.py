import math
import pathlib
import tempfile
import typing

import einops
import huggingface_hub as hf
import pydantic as pyd
import safetensors.torch
import torch
import tqdm


class IDiTConfig(pyd.BaseModel):
    input_height: int
    input_width: int
    input_dimension: int
    hidden_dimension: int
    head_dimension: int
    condition_dimension: int
    frequency_dimension: int
    layers: int
    iterations: int


class ConditionEmbedding(torch.nn.Module):  # https://arxiv.org/abs/2006.10739
    def __init__(self, condition_dimension: int, frequency_dimension: int, max_frequency: float = 100.0) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(frequency_dimension, condition_dimension, bias=False)
        self.frequency: torch.Tensor

        self.register_buffer("frequency", torch.linspace(math.log(1.0), math.log(max_frequency), steps=frequency_dimension // 2).exp())

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x = condition.unsqueeze(-1) * self.frequency
        x = self.linear(torch.cat([x.cos(), x.sin()], dim=-1))

        return x


class PatchEmbedding(torch.nn.Module):  # https://arxiv.org/abs/2010.11929
    def __init__(self, input_dimension: int, hidden_dimension: int, height: int, width: int) -> None:
        super().__init__()

        self.height = height
        self.width = width
        self.linear = torch.nn.Linear(input_dimension, hidden_dimension, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(einops.rearrange(x, "b c h w -> b (h w) c", h=self.height, w=self.width))


class PatchUnembedding(torch.nn.Module):  # https://arxiv.org/abs/2010.11929
    def __init__(self, hidden_dimension: int, input_dimension: int, height: int, width: int) -> None:
        super().__init__()

        self.height = height
        self.width = width
        self.linear = zero_init(torch.nn.Linear(hidden_dimension, input_dimension, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(self.linear(x), "b (h w) c -> b c h w", h=self.height, w=self.width)


class AdaRMSNorm(torch.nn.Module):  # https://arxiv.org/abs/2401.11605
    def __init__(self, hidden_dimension: int, condition_dimension: int) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(condition_dimension, hidden_dimension * 2, bias=False)
        self.norm = torch.nn.RMSNorm(hidden_dimension, elementwise_affine=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(condition).unsqueeze(1).chunk(2, dim=-1)

        return self.norm(x) * (1 + scale) + shift


class PostRMSNorm(torch.nn.Module):  # https://arxiv.org/abs/2601.19895
    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()

        self.norm = torch.nn.RMSNorm(hidden_dimension)

    def forward(self, x: torch.Tensor, y: torch.Tensor, layers: int) -> torch.Tensor:
        return self.norm(layers * x + y)


class MLP(torch.nn.Module):  # https://arxiv.org/abs/2212.09748
    def __init__(self, hidden_dimension: int, condition_dimension: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(hidden_dimension, hidden_dimension * 4, bias=False)
        self.linear_2 = zero_init(torch.nn.Linear(hidden_dimension * 4, hidden_dimension, bias=False))
        # self.ada_norm = AdaRMSNorm(hidden_dimension, condition_dimension)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x = self.ada_norm(x, condition)
        x = self.linear_1(x)
        x = self.linear_2(torch.nn.functional.silu(x))

        return x


class Attention(torch.nn.Module):  # https://arxiv.org/abs/2212.09748
    def __init__(self, hidden_dimension: int, head_dimension: int, condition_dimension: int) -> None:
        super().__init__()

        self.head_dimension = head_dimension
        self.linear_1 = torch.nn.Linear(hidden_dimension, hidden_dimension * 3, bias=False)
        self.linear_2 = zero_init(torch.nn.Linear(hidden_dimension, hidden_dimension, bias=False))
        self.ada_norm = AdaRMSNorm(hidden_dimension, condition_dimension)
        self.qk_norm = torch.nn.RMSNorm(head_dimension)  # https://arxiv.org/abs/2010.04245

    def forward(self, x: torch.Tensor, condition: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        x = self.ada_norm(x, condition)
        q, k, v = einops.rearrange(self.linear_1(x), "b l (n h d) -> n b h l d", n=3, d=self.head_dimension)
        q = apply_rope(self.qk_norm(q), rope)
        k = apply_rope(self.qk_norm(k), rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = self.linear_2(einops.rearrange(x, "b h l d -> b l (h d)"))

        return x


class IDiTBlock(torch.nn.Module):
    def __init__(self, config: IDiTConfig) -> None:
        super().__init__()

        self.attention = Attention(config.hidden_dimension, config.head_dimension, config.condition_dimension)
        self.mlp = MLP(config.hidden_dimension, config.condition_dimension)
        self.keel_1 = PostRMSNorm(config.hidden_dimension)
        self.keel_2 = PostRMSNorm(config.hidden_dimension)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, rope: torch.Tensor, layers: int) -> torch.Tensor:
        x = self.keel_1(x, self.attention(x, condition, rope), layers)
        x = self.keel_2(x, self.mlp(x, condition), layers)

        return x


class IDiT(torch.nn.Module):
    def __init__(self, config: IDiTConfig) -> None:
        super().__init__()

        self.config = config
        self.time_embedding = ConditionEmbedding(config.condition_dimension, config.frequency_dimension)
        self.iteration_embedding = ConditionEmbedding(config.condition_dimension, config.frequency_dimension)
        self.patch_embedding = PatchEmbedding(config.input_dimension, config.hidden_dimension, config.input_height, config.input_width)
        self.patch_unembedding = PatchUnembedding(config.hidden_dimension, config.input_dimension, config.input_height, config.input_width)
        self.blocks = torch.nn.ModuleList([IDiTBlock(config) for _ in range(config.layers)])
        self.rope: torch.Tensor

        self.register_buffer("rope", create_rope_2d(config.head_dimension, config.input_height, config.input_width))

    def predict(self, x: torch.Tensor, *, time: torch.Tensor) -> torch.Tensor:
        time_condition = self.time_embedding(time)
        x = self.patch_embedding(x)

        for iteration in range(self.config.iterations):
            iteration_condition = self.iteration_embedding(torch.full((x.size(0),), iteration / max(self.config.iterations, 1), device=x.device, dtype=x.dtype))
            condition = time_condition + iteration_condition

            for block in self.blocks:
                x = block(x, condition, self.rope, layers=self.config.layers * self.config.iterations)

        x = self.patch_unembedding(x)

        return x

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        time = torch.randn(data.size(0), 1, 1, 1, device=data.device, dtype=data.dtype).sigmoid()
        noise = torch.randn_like(data)
        noisy = (1 - time) * data + time * noise
        prediction = self.predict(noisy, time=time.view(-1))
        loss = torch.nn.functional.mse_loss(prediction.float(), (noise - data).float())

        return loss

    @classmethod
    def from_pretrained(cls, path: str, revision: str | None = None) -> typing.Self:
        checkpoint_path = hf.snapshot_download(repo_id=path, revision=revision)
        checkpoint_path = pathlib.Path(checkpoint_path)
        config = IDiTConfig.model_validate_json((checkpoint_path / "config.json").read_text())
        model = cls(config)
        model.load_state_dict(safetensors.torch.load_file(checkpoint_path / "model.safetensors"))
        tqdm.tqdm._instances.clear()  #  type: ignore

        return model

    def push_to_hub(self, path: str, private: bool = True) -> None:
        hf.create_repo(path, private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            (checkpoint_path / "config.json").write_text(self.config.model_dump_json(indent=4))
            safetensors.torch.save_file(self.state_dict(), checkpoint_path / "model.safetensors")
            hf.upload_folder(repo_id=path, folder_path=str(checkpoint_path))


# Functions.


def create_rope_1d(head_dimension: int, length: int) -> torch.Tensor:  # https://arxiv.org/abs/2104.09864
    frequency = torch.linspace(math.log(1.0), math.log(length / 4), steps=head_dimension // 2).exp()
    x = torch.linspace(-math.pi / 2, math.pi / 2, length)
    x = x.unsqueeze(-1) * frequency
    x = torch.stack([x.cos(), x.sin()]).repeat_interleave(2, dim=-1)

    return x


def create_rope_2d(head_dimension: int, height: int, width: int) -> torch.Tensor:  # https://arxiv.org/abs/2401.11605
    rope_height = create_rope_1d(head_dimension // 2, height).unsqueeze(2).repeat_interleave(width, dim=2)
    rope_width = create_rope_1d(head_dimension // 2, width).unsqueeze(1).repeat_interleave(height, dim=1)

    return torch.cat([rope_height, rope_width], dim=-1).view(2, -1, head_dimension)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    x_hat = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)

    return x * rope[0] + x_hat * rope[1]


T = typing.TypeVar("T", bound=torch.nn.Module)


def zero_init(module: T) -> T:
    for parameter in module.parameters():
        torch.nn.init.zeros_(parameter)

    return module
