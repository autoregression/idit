"""Tests for IDiT configuration system."""

import os
import pytest
from idit.config import (
    DATASET_CONFIGS,
    IDiTPresets,
    IDiTTrainerConfig,
    IDiTSamplerConfig,
)


class TestDatasetConfigs:
    def test_bitmind_ffhq_256_config(self):
        config = DATASET_CONFIGS["bitmind/ffhq-256"]
        assert config["resolution"] == 256
        assert config["steps"] == 100_000
        assert config["patch_size"] == 16
        assert config["input_dimension"] == 3

    def test_mnist_config(self):
        config = DATASET_CONFIGS["mnist"]
        assert config["resolution"] == 28
        assert config["steps"] == 20_000
        assert config["patch_size"] == 2
        assert config["input_dimension"] == 1


class TestIDiTPresetsDefaults:
    def test_default_values(self):
        presets = IDiTPresets()
        assert presets.dataset_path == "bitmind/ffhq-256"
        assert presets.seed == 0
        assert presets.resolution == 256
        assert presets.steps == 100_000
        assert presets.batch_size == 4
        assert presets.patch_size == 16
        assert presets.input_dimension == 3

    def test_with_dataset_applies_overrides(self):
        presets = IDiTPresets()
        merged = presets.with_dataset()
        assert merged.resolution == 256
        assert merged.patch_size == 16
        assert merged.steps == 100_000


class TestIDiTTrainerConfig:
    def test_from_presets_default(self):
        presets = IDiTPresets().with_dataset()
        config = IDiTTrainerConfig.from_presets(presets)
        assert config.seed == 0
        assert config.dataset_path == "bitmind/ffhq-256"
        assert config.resolution == 256
        assert config.patch_size == 16

    def test_from_presets_custom(self):
        presets = IDiTPresets(patch_size=8, learning_rate=1e-4)
        merged = presets.with_dataset()
        config = IDiTTrainerConfig.from_presets(merged)
        assert config.patch_size == 8
        assert config.learning_rate == 1e-4


class TestIDiTSamplerConfig:
    def test_from_presets_default(self):
        presets = IDiTPresets().with_dataset()
        config = IDiTSamplerConfig.from_presets(presets)
        assert config.seed == 0
        assert config.resolution == 256
        assert config.steps == 20

    def test_from_presets_custom_inference_steps(self):
        presets = IDiTPresets(inference_steps=50, inference_batch_size=32)
        merged = presets.with_dataset()
        config = IDiTSamplerConfig.from_presets(merged)
        assert config.steps == 50
        assert config.batch_size == 32


class TestEnvironmentVariables:
    def test_env_var_patch_size(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {**os.environ, "PATCH_SIZE": "8"})
        presets = IDiTPresets(_env_file=None)  # type: ignore
        merged = presets.with_dataset()
        assert merged.patch_size == 8

    def test_env_var_resolution(self, monkeypatch):
        monkeypatch.setenv("RESOLUTION", "512")
        presets = IDiTPresets(_env_file=None)  # type: ignore
        merged = presets.with_dataset()
        assert merged.resolution == 512

    def test_env_var_learning_rate(self, monkeypatch):
        monkeypatch.setenv("LEARNING_RATE", "0.0005")
        presets = IDiTPresets(_env_file=None)  # type: ignore
        config = IDiTTrainerConfig.from_presets(presets.with_dataset())
        assert config.learning_rate == 5e-4

    def test_env_var_inference_steps(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_STEPS", "100")
        presets = IDiTPresets(_env_file=None)  # type: ignore
        merged = presets.with_dataset()
        config = IDiTSamplerConfig.from_presets(merged)
        assert config.steps == 100

    def test_env_var_seed(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {**os.environ, "SEED": "42"})
        presets = IDiTPresets(_env_file=None)  # type: ignore
        merged = presets.with_dataset()
        config = IDiTTrainerConfig.from_presets(merged)
        assert config.seed == 42

    def test_env_var_batch_size(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {**os.environ, "BATCH_SIZE": "8"})
        presets = IDiTPresets(_env_file=None)  # type: ignore
        merged = presets.with_dataset()
        assert merged.batch_size == 8


class TestDatasetOverridesWithEnvVars:
    def test_env_var_overrides_dataset_config(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {**os.environ, "PATCH_SIZE": "4"})
        presets = IDiTPresets(dataset_path="mnist")
        merged = presets.with_dataset()
        assert merged.resolution == 28
        assert merged.patch_size == 4

    def test_default_mnist_config(self):
        presets = IDiTPresets(dataset_path="mnist").with_dataset()
        config = IDiTTrainerConfig.from_presets(presets)
        assert config.resolution == 28
        assert config.input_dimension == 1
