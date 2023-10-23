import os
# import argparse
# import logging
# from tqdm import tqdm

# import torch
import torchvision.transforms as T
# from torch.utils.data import DataLoader, ConcatDataset

# from models.model import TDClassifier
from models.model import LitTDClassifier
from models.model import LitTDDataModule
from models.model import TDLightningCLI
# from dataset import PatchDataset

import pickle
import mlflow.pytorch
# from mlflow.models import infer_signature

# import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def cli_main():
  early_stopping = EarlyStopping(
    monitor="val_loss",
  )

  if os.path.join(os.getcwd(), "results") is False:
    os.makedirs(os.path.join(os.getcwd(), "results"))
  checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(os.getcwd(), "results"), save_top_k=1, verbose=True, monitor="val_loss", mode="min"
  )

  lr_logger = LearningRateMonitor()
  cli = TDLightningCLI(
    LitTDClassifier,
    LitTDDataModule,
    run=False,
    save_config_callback=None,
    trainer_defaults={"callbacks": [early_stopping, checkpoint_callback, lr_logger]},
  )

  if cli.trainer.global_rank == 0:
      mlflow.pytorch.autolog()
  # cli.model=torch.compile(cli.model)
  cli.trainer.fit(cli.model, datamodule=cli.datamodule)
  cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__=="__main__":
  # main()
  cli_main()

# 3545833472
# find . -type d | xargs -n1 -I{} rmdir {}
'''20231019-Thyroid
[v0.1]
mlflow run . -P mode=train -P cancer_type=thyroid -P backbone=resnext50_32x4d \
-P epoch=50 -P bsize=32 -P num_class=3 \
-P train_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 \
-P infer_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 

mlflow run . -P mode=test -P cancer_type=thyroid -P backbone=resnext50_32x4d \
-P epoch=50 -P bsize=32 -P num_class=3 \
-P train_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 \
-P infer_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 

[v0.2]
mlflow run . -P backbone=resnext50_32x4d \
-P max_epochs=100 -P batch_size=128 -P num_class=2 \
-P dataset_dir=/data/rnd1712/dataset/thyroid/classification/v0-2 &> /dev/null &

mlflow run . -P mode=test -P cancer_type=thyroid -P backbone=resnext50_32x4d \
-P bsize=64 -P num_class=2 \
-P train_dir=/data/rnd1712/dataset/thyroid/classification/v0-2 \
-P infer_dir=/data/rnd1712/dataset/thyroid/classification/v0-2

mlflow run . -P backbone=resnext50_32x4d \
-P max_epochs=5 -P batch_size=128 -P num_class=2 \
-P dataset_dir=/data/rnd1712/dataset/thyroid/classification/v0-2
'''


# 34423 (mlflow ui)