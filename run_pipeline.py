from src.pipeline.processor import Processor
from src.utils.config_loader import load_yaml_config

config = load_yaml_config("configs/pipeline_config.yaml")
processor = Processor()
results = processor.run_pipeline(config)
