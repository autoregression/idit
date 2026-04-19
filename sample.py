from idit.config import IDiTPresets, IDiTSamplerConfig
from idit.sampler import IDiTSampler

presets = IDiTPresets()
merged = presets.with_dataset()
sampler = IDiTSampler(config=IDiTSamplerConfig.from_presets(merged))
samples = sampler.sample()
