import yaml
from schema import Schema, SchemaError

def is_valid(conf_yaml):
  config_schema = Schema({
    "common": {
      "version": str,
      "infer_version": str,
      "cancer": str,
      "type": str,
      "save_dir": str,
    },
    "training": {
      "n_epochs": int,
      "learning_rate": float,
      "weight_decay": float,
      "batch_size": int,
    },
    "model": {
      "pretrained": bool,
      "backbone": str,
      "n_classes": int,
    },
    "dataset": {
      "train": str,
      "infer": str,
    }
  })

  try:
    config_schema.validate(conf_yaml)
    print("Configuration is valid")
  except SchemaError as se:
    raise se
