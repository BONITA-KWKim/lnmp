import os
# import argparse
# import logging
# from tqdm import tqdm

# import torch
import torchvision.transforms as T
# from torch.utils.data import DataLoader, ConcatDataset

# from models.model import TDClassifier
from models.model import LitTDClassifier
from dataset import LitTDDataModule
from models.model import TDLightningCLI
# from dataset import PatchDataset

# import pickle
import mlflow.pytorch
# from mlflow.models import infer_signature

# import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


# import datetime
from datetime import datetime
def cli_main():
  early_stopping = EarlyStopping(
    monitor="val_loss", patience=10
  )

  date_ = datetime.now().strftime('%Y%m%d-%H%M')
  model_saved_dir = os.path.join(os.getcwd(), "results", date_)
  if os.path.join(model_saved_dir) is False:
    os.makedirs(model_saved_dir)
  checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(model_saved_dir), filename="best", save_top_k=1, 
    verbose=False, monitor="val_loss", mode="min"
  )

  lr_logger = LearningRateMonitor()
  cli = TDLightningCLI(
    LitTDClassifier,
    LitTDDataModule,
    run=False,
    save_config_callback=None,
    trainer_defaults={
      "callbacks": [early_stopping, checkpoint_callback, lr_logger]
    },
  )
  # set run name
  mlflow.set_tag("mlflow.runName", "Thyroid-V0.1-Detection-"+date_) 

  if cli.trainer.global_rank == 0:
      mlflow.pytorch.autolog()
  # cli.model=torch.compile(cli.model)
  cli.trainer.fit(cli.model, datamodule=cli.datamodule)
  # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
  cli.trainer.test(ckpt_path=os.path.join(model_saved_dir, 'best.ckpt'), datamodule=cli.datamodule)

  """Predict
  model = LightningTransformer.load_from_checkpoint(PATH)
  dataset = WikiText2()
  test_dataloader = DataLoader(dataset)
  trainer = pl.Trainer()
  predictions = trainer.predict(model, dataloaders=test_dataloader)
  """

if __name__=="__main__":
  # main()
  cli_main()

# 3545833472
# find . -type d | xargs -n1 -I{} rmdir {}
'''20231019-Thyroid
[v0.1]
mlflow run . -P backbone=resnext50_32x4d \
-P max_epochs=100 -P batch_size=32 -P num_class=3 \
-P dataset_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 &> logs/20231024N001-V0_1.log &
tail -f logs/20231024N001-V0_1.log

mlflow run . -P mode=test -P cancer_type=thyroid -P backbone=resnext50_32x4d \
-P epoch=50 -P bsize=32 -P num_class=3 \
-P train_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 \
-P infer_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 

[v0.2]
mlflow run . -P backbone=resnext50_32x4d \
-P max_epochs=2 -P batch_size=64 -P num_class=2 \
-P dataset_dir=/data/rnd1712/dataset/thyroid/classification/v0-2 &> logs/20231024N005-V0_2.log &

mlflow run . -P mode=test -P cancer_type=thyroid -P backbone=resnext50_32x4d \
-P bsize=64 -P num_class=2 \
-P train_dir=/data/rnd1712/dataset/thyroid/classification/v0-2 \
-P infer_dir=/data/rnd1712/dataset/thyroid/classification/v0-2

mlflow run . -P backbone=resnext50_32x4d \
-P max_epochs=5 -P batch_size=128 -P num_class=2 \
-P dataset_dir=/data/rnd1712/dataset/thyroid/classification/v0-2
'''


# 34423 (mlflow ui)
