from src.idit.sampler import IDiTSampler, IDiTSamplerConfig

sampler = IDiTSampler(config=IDiTSamplerConfig())
samples = sampler.sample()
