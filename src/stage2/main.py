import os
import random
import warnings
import argparse
import logging

import numpy as np
import torch

from src.stage2.models.mil_net import MILNetImageOnly, MILNetWithClinicalData
from src.stage2.models.backbones.backbone_builder import BACKBONES
from src.stage2.utils.utils import *
from src.stage2.dataset import MILDataset


def parser_args():
  parser = argparse.ArgumentParser()

  # dataset
  parser.add_argument("--train_json_path", required=True)
  parser.add_argument("--val_json_path", required=True)
  parser.add_argument("--test_json_path", required=True)
  parser.add_argument("--data_dir_path", default="./dataset")
  parser.add_argument("--clinical_data_path")
  parser.add_argument("--preloading", action="store_true")
  parser.add_argument("--num_classes", type=int, choices=[2, 3], default=2)

  # model
  parser.add_argument("--backbone", choices=BACKBONES, default="vgg16_bn")

  # optimizer
  parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="SGD")
  parser.add_argument("--lr", type=float, default=0.0001)
  parser.add_argument("--momentum", type=float, default=0.3)
  parser.add_argument("--weight_decay", type=float, default=0.001)
  parser.add_argument("--l1_weight", type=float, default=0)

  # output
  parser.add_argument("--log_dir_path", default="./logs")
  parser.add_argument("--log_name", required=True)
  parser.add_argument("--save_epoch_interval", type=int, default=10)

  # other
  parser.add_argument("--mode", type=str, choices=["train", "test"],
                      default='train')
  parser.add_argument("--epoch", type=int, default=20)
  parser.add_argument("--train_stop_auc", type=float, default=0.99)
  parser.add_argument("--merge_method", choices=["max", "mean", "not_use"], 
                      default="mean")
  parser.add_argument("--seed", type=int, default=1121)
  parser.add_argument("--num_workers", type=int, default=4)

  args = parser.parse_args()

  return args


def init_logger(log_lv=logging.DEBUG, log_file='log/default.log'):
  logger = logging.getLogger('mil_logger')
  logger.setLevel(log_lv)

  ch = logging.StreamHandler()
  ch.setLevel(log_lv)

  fh = logging.FileHandler(log_file)
  fh.setLevel(log_lv)

  # create formatter
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  # add formatter to ch
  ch.setFormatter(formatter)
  fh.setFormatter(formatter)

  # add ch to logger
  logger.addHandler(ch)
  logger.addHandler(fh)

  return logger

def init_output_directory(log_dir_path, log_name):
  checkpoint_path = os.path.join(log_dir_path, log_name, "checkpoint")
  os.makedirs(checkpoint_path, exist_ok=True)

  return checkpoint_path


def seed_everything(seed):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def init_optimizer(args, model):
  if args.optimizer == "Adam":
    return torch.optim.Adam(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
  elif args.optimizer == "SGD":
    return torch.optim.SGD(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
      momentum=args.momentum
    )
  else:
    raise NotImplementedError


def init_dataloader(args):
  train_dataset = MILDataset(
    args.train_json_path, args.data_dir_path, args.clinical_data_path, 
    is_preloading=args.preloading
  )
  train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True, 
    num_workers=args.num_workers
  )

  val_dataset = MILDataset(
    args.val_json_path, args.data_dir_path, args.clinical_data_path, 
    is_preloading=args.preloading
  )
  val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=1, shuffle=False, pin_memory=True, 
    num_workers=args.num_workers
  )

  test_dataset = MILDataset(
    args.test_json_path, args.data_dir_path, args.clinical_data_path, 
    is_preloading=args.preloading
  )
  test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False, pin_memory=True, 
    num_workers=args.num_workers
  )

  return train_loader, val_loader, test_loader


if __name__ == "__main__":
  args = parser_args()

  # init setting
  warnings.filterwarnings("ignore")
  checkpoint_path = init_output_directory(args.log_dir_path, args.log_name)
  seed_everything(args.seed)

  # init logger
  logger = init_logger(log_file=os.path.join(args.log_dir_path, args.log_name+'.log'))

  # init dataloader
  train_loader, val_loader, test_loader = init_dataloader(args)

  # init model
  if args.clinical_data_path:
    model = MILNetWithClinicalData(
      num_classes=args.num_classes, backbone_name=args.backbone, 
      clinical_data_size=5, expand_times=10
    )
  else:
    model = MILNetImageOnly(
      num_classes=args.num_classes, backbone_name=args.backbone
    )

  # load model
  last_model_path = os.path.join(checkpoint_path, "last.pth")
  if os.path.exists(last_model_path):
    model = load_checkpoint(model, last_model_path)
  model = model.cuda()

  # init optimizer and lr scheduler
  optimizer = init_optimizer(args, model)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer=optimizer, T_0=20, T_mult=2
  )

  # init training function
  if args.num_classes > 2:
    # print("multiple classification")
    logger.info("multiple classification")
    main_fun = train_val_test_multi_class
  else:
    # print("binary classification")
    logger.info("binary classification")
    main_fun = train_val_test_binary_class

  # training
  if args.mode=='test':
    """TEST MODE
    """
    model = load_checkpoint(model, os.path.join(checkpoint_path, "last.pth"))
    main_fun("test", 0, model, val_loader, None, args.merge_method)
    # print("END")
    logger.info("END")
  elif args.mode=='train':
    """TRAIN MODE
    """
    best_auc = 0
    best_epoch = 0
    for epoch in range(1, args.epoch + 1):
      scheduler.step()
      
      # TRAIN
      train_auc = main_fun(
        "train", epoch, model, train_loader, optimizer, args.merge_method, logger
      )
      # VAL
      val_auc = main_fun(
        "val", epoch, model, val_loader, None, args.merge_method, logger
      )
      # TEST
      main_fun(
        "test", epoch, model, test_loader, None, args.merge_method, logger
      )
              
      # save best
      if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch
        save_checkpoint(model, os.path.join(checkpoint_path, "best.pth"))

      # save model
      if epoch % args.save_epoch_interval == 0:
        save_checkpoint(model, os.path.join(checkpoint_path, f"{epoch}.pth"))
        save_checkpoint(model, os.path.join(checkpoint_path, f"last.pth"))

      # print("-" * 120)
      logger.info(str("-" * 120))

      # early stopping
      if train_auc > args.train_stop_auc and (epoch > 3):
        # print(f"early stopping, epoch: {epoch}, train_auc: {train_auc:.3f} (>{args.train_stop_auc})")
        logger.info(f"early stopping, epoch: {epoch}, train_auc: {train_auc:.3f} (>{args.train_stop_auc})")
        break
      
      torch.cuda.empty_cache()

    # print(f"end, best_auc: {best_auc}, best_epoch: {best_epoch}")
    logger.info(f"end, best_auc: {best_auc}, best_epoch: {best_epoch}")

'''

CUDA_VISIBLE_DEVICES=1 python main.py --epoch 1 \
--train_json_path dataset/ovarian/cohort/test-jsons-holdout-b50/train-type-001.json \
--val_json_path dataset/ovarian/cohort/test-jsons-holdout-b50/val-type-001.json \
--test_json_path dataset/ovarian/cohort/test-jsons-holdout-b50/test-type-001.json \
--num_classes 2 \
--backbone vgg16_bn \
--log_name vgg16_bn_holdout_b50_testN001

'''