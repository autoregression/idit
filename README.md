# IDiT - Iterative Diffusion Transformer

A DiT architecture that uses each layer more than once, ala [Universal Transformers](https://arxiv.org/abs/1807.03819).

## Installation

```
git clone https://github.com/autoregression/idit
```

```
uv sync
```

## Usage

Train (~5m on an M4).

```
uv run train.py 
```

Sample.

```
uv run sample.py
```

Tweak parameters.

```
steps=20000 batch_size=16 uv run train.py
```

## Configuration

### Trainer Configuration


| Parameter               | Description                                                        | Default   |
| ----------------------- | ------------------------------------------------------------------ | --------- |
| `seed`                  | The RNG seed.                                                      | `0`       |
| `dataset_path`          | The path to a Hugging Face dataset (e.g., `"ILSVRC/imagenet-1k"`). | `"mnist"` |
| `split`                 | The dataset split to train on, typically `"train"`.                | `"train"` |
| `column`                | The dataset column to train on, typically `"image"`.               | `"image"` |
| `resolution`            | The resolution to train on.                                        | `28`      |
| `input_dimension`       | The number of image channels (e.g., 1 for grayscale, 3 for RGB).   | `1`       |
| `hidden_dimension`      | The dimension of the model.                                        | `64`      |
| `head_dimension`        | The dimension of attention heads.                                  | `16`      |
| `condition_dimension`   | The dimension of condition embeddings.                             | `64`      |
| `frequency_dimension`   | The dimension of frequency embeddings.                             | `256`     |
| `layers`                | The number of layers.                                              | `2`       |
| `iterations`            | The number of iterations.                                          | `4`       |
| `steps`                 | The number of training steps.                                      | `10000`   |
| `batch_size`            | The training batch size.                                           | `4`       |
| `gradient_accumulation` | The number of gradient accumulation steps.                         | `1`       |
| `learning_rate`         | The learning rate.                                                 | `1e-3`    |
| `warmup`                | The number of warmup steps.                                        | `100`     |
| `cooldown`              | The number of cooldwon steps.                                      | `500`     |


### Sampler Configuration


| Parameter         | Description                        | Default        |
| ----------------- | ---------------------------------- | -------------- |
| `seed`            | The RNG seed.                      | `0`            |
| `resolution`      | The resolution to generate at.     | `28`           |
| `steps`           | The number of solver steps.        | `20`           |
| `batch_size`      | The number of samples to generate. | `16`           |
| `samples_path`    | The path to save samples to.       | `"samples"`    |
| `checkpoint_path` | The path to the model.             | `"checkpoint"` |


