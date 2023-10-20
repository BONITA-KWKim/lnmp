import os
import argparse
import logging
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset

# from models.model import TDClassifier
from models.model import LitTDClassifier
from dataset import PatchDataset

import pickle
import mlflow.pytorch
# from mlflow.models import infer_signature

import lightning.pytorch as pl


def get_logger(log_lv):
  log_level = {
    'fatal': logging.FATAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
  }
  # create logger
  logger = logging.getLogger('LNMP logger')
  logger.setLevel(log_level[log_lv])

  # create console handler and set level to debug
  ch = logging.StreamHandler()
  ch.setLevel(log_level[log_lv])

  # create formatter
  formatter = logging.Formatter('[%(levelname)s]::(%(asctime)s) %(name)s - %(message)s')

  # add formatter to ch
  ch.setFormatter(formatter)

  # add ch to logger
  logger.addHandler(ch)

  return logger


'''options
'''
def init_arg():
  parser = argparse.ArgumentParser(description='WSI patching process.')

  parser.add_argument('--verbose', default='info', choices=['info', 'debug'], 
                      type=str, help='The level of log')
  # common
  parser.add_argument('--mode', default='train', choices=['train', 'test'], 
                      type=str, help='Train/Test')
  parser.add_argument('--cancer_type', default='breast', type=str,
                      help='Cancer type to detect tumor')
  parser.add_argument('--bsize', default=16, type=int,
                      help='The number of batch size')
  parser.add_argument('--num_worker', default=4, type=int,
                      help='The number of wokers')
  parser.add_argument('--epoch', default=10, type=int,
                      help='The number of epochs')
  # model 
  parser.add_argument('--backbone', default='resnext50_32x4d', type=str,
                      help='Backbone model')
  parser.add_argument('--num_class', default=3, type=int,
                      help='The number of classes')
  parser.add_argument('--lr', default=8e-4, type=float,
                      help='Learning rate')
  parser.add_argument('--decay', default=3e-5, type=float,
                      help='Weight Decay')
  parser.add_argument('--pretrained', default=True, type=bool,
                      help='Using pretrained model')
  # Dataset directories
  parser.add_argument('--train_dir', default='./dataset/train', type=str,
                      help='Train dataset directory')
  parser.add_argument('--infer_dir', default='./dataset/infer', type=str,
                      help='Inference dataset directory')
  parser.add_argument('--save_dir', default='./results', type=str,
                      help='Directory to save results')

  args = parser.parse_args()
  return args


def get_dataloader(logger, num_class, dataset_dir, bsize, num_workers=4):
  train_dataset = None
  val_dataset = None
  test_dataset = None
  datasets = {
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset,
  }
  for dataset_type in ["train", "val", "test"]:
    _list = []
    for label in range(num_class):
      path = os.path.join(dataset_dir, str(label), dataset_type)
      _dataset = PatchDataset(path, label, 
                                    num_classes=num_class, 
                                    one_hot=True)
      _list.append(_dataset)
    datasets[dataset_type] = ConcatDataset(_list)

  # debug
  for dataset_type in ["train", "val", "test"]:
    logger.debug(f"{dataset_type} dataset: {len(datasets[dataset_type])}")
  
  t_dataloader = DataLoader(datasets["train"], batch_size=bsize, shuffle=True, 
    drop_last=True, num_workers=num_workers)
  v_dataloader = DataLoader(datasets["val"], batch_size=bsize, shuffle=False, 
    drop_last=False, num_workers=num_workers)
  tst_dataloader = DataLoader(datasets["test"], batch_size=bsize, shuffle=False, 
    drop_last=False, num_workers=num_workers)
  
  return t_dataloader, v_dataloader, tst_dataloader


# def train_e(logger, model, dataloader):
#   # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
#   trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
#   trainer.fit(model=model, train_dataloaders=dataloader)


def train_epoch(logger, model, device, dataloader, loss_fn, optimizer):
  t_loss, correct= .0, 0
  model.train()

  for images, labels in dataloader:
    images, labels = images.to(device), labels.to(device)

    train_transforms = T.RandomApply(
      [
        T.RandomRotation((-270, 270)),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
        T.RandomHorizontalFlip(p=0.7),
        T.RandomVerticalFlip(p=0.7),
        T.RandomGrayscale(),
        T.RandomErasing()
      ]
    )

    # to expand
    for _ in range(3):
      optimizer.zero_grad()
      
      output = model(train_transforms(images))
      loss = loss_fn(output, labels)
      loss.backward()
      optimizer.step()

      t_loss += loss.item() / images.size(0)
      
      pred_labels = torch.argmax(output, 1)
      t_labels = torch.argmax(labels, -1)
      correct += (pred_labels == t_labels).sum().item()
  return t_loss, correct


def valid_epoch(logger, model, device, dataloader, loss_fn):
  v_loss, correct = .0, 0
  model.eval()

  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)

      output = model(images)
      # results
      loss = loss_fn(output, labels)
      v_loss += loss.item() / images.size(0)

      pred_labels = torch.argmax(output, -1)
      t_labels = torch.argmax(labels, -1)
      correct += (pred_labels == t_labels).sum().item()
  return v_loss, correct


def main():
  args = init_arg()
  logger = get_logger(args.verbose)
  logger.info("START")

  '''load dataset
  '''
  # t_dataloader, v_dataloader, _ = get_dataloader(conf_yaml)
  t_dataloader, v_dataloader, tst_dataloader = get_dataloader(
    logger, args.num_class, args.train_dir, args.bsize)

  with mlflow.start_run() as run:
    '''MLflow
    '''
    mlflow.set_tag("mlflow.runName", "Lightning Test") # set run name

    '''Model
    ''' 
    # torch.autograd.set_detect_anomaly(True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logger.debug(f'device : {device}')
    # logger.debug(f'Current Cuda Device : {torch.cuda.current_device()}')

    '''Training
    '''
    # saving_dir = args.save_dir
    # cancer_type = args.cancer_type

    # if False == os.path.exists(saving_dir):
      # os.makedirs(saving_dir)
    
    model = LitTDClassifier(args.num_class, args.backbone, args.pretrained)
    
    mlflow.pytorch.autolog()
    
    if "test"==args.mode:
      # Inference
      pass
    elif "train"==args.mode:
      # train the model
      trainer = pl.Trainer(
        devices=3, accelerator="auto", check_val_every_n_epoch=5,
        limit_train_batches=args.bsize, max_epochs=args.epoch
      )
      trainer.fit(model=model, train_dataloaders=t_dataloader, 
                  val_dataloaders=v_dataloader)
      
      # validate
      # trainer.validate(model=model, dataloaders=v_dataloader)

  logger.info("END")


if __name__=="__main__":
  main()

'''20231019-Thyroid
[v0.1]
mlflow run . -P mode=train -P cancer_type=thyroid -P backbone=resnext50_32x4d \
-P epoch=50 -P bsize=32 -P num_class=3 \
-P train_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 \
-P infer_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 \
-P save_dir=./result-thyroid-v0_1-20231020N001

mlflow run . -P mode=test -P cancer_type=thyroid -P backbone=resnext50_32x4d \
-P epoch=50 -P bsize=32 -P num_class=3 \
-P train_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 \
-P infer_dir=/data/rnd1712/dataset/thyroid/classification/v0-1 \
-P save_dir=./result-thyroid-v0_1-20231019N001

[v0.2]
mlflow run . -P mode=train -P cancer_type=thyroid -P backbone=resnext50_32x4d \
-P epoch=50 -P bsize=64 -P num_class=2 \
-P train_dir=/data/rnd1712/dataset/thyroid/classification/v0-2 \
-P infer_dir=/data/rnd1712/dataset/thyroid/classification/v0-2 \
-P save_dir=./result-thyroid-v0_2-20231020N001
'''
