from idit.config import IDiTPresets, IDiTTrainerConfig
from idit.trainer import IDiTTrainer

presets = IDiTPresets()
merged = presets.with_dataset()
trainer = IDiTTrainer(config=IDiTTrainerConfig.from_presets(merged))
trainer.train()
