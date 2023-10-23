import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

# import lightning.pytorch as pl
import lightning as L
from lightning.pytorch.cli import LightningCLI

from dataset import PatchDataset


def get_model_by_name(modelname, pretrained):
  model = torch.hub.load('pytorch/vision:v0.12.0', modelname, 
                         pretrained=pretrained)
  feature_size = 1000
  return model, feature_size


class TDClassifier(nn.Module):
  def __init__(self, num_classes, backbone='resnext50_32x4d', pretrained=True):
    super(TDClassifier, self).__init__()
    self.cnn, feature_size = get_model_by_name(backbone, pretrained)
    self.classifier = nn.Sequential(
      *(self._get_fcn(feature_size, num_classes, 10))
    )

  def _get_fcn(self, input_size, output_size, depth):
    module_list = []
    module_list += [nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2), ]
    for _ in range(depth-3):
      module_list += [nn.Linear(512, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(),
              nn.Dropout(0.2)]
    module_list += [nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size), ]
    return module_list

  def forward(self, x):
    output = self.cnn(x)
    output = self.classifier(output)
    output = nn.functional.softmax(output, 1)
    return output

  def load_cnn_dict(self, pretrained_dict):
    self.cnn.load_state_dict(pretrained_dict)


class LitTDDataModule(L.LightningDataModule):
  def __init__(self, dataset_dir, num_class, batch_size, num_workers):
    """
    Initialization of inherited lightning data module
    """
    super(LitTDDataModule, self).__init__()
    self.dataset_dir = dataset_dir
    self.train_dataset = None
    self.val_dataset = None
    self.test_dataset = None
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.num_class = num_class
    # self.num_class = len(os.listdir(dataset_dir))
    self.train_count = None
    self.val_count = None
    self.test_count = None
    self.RANDOM_SEED = 42

  def setup(self, stage=None):
    """
    Split the data into train, test, validation data

    :param stage: Stage - training or testing
    """
    if stage=="fit":
      for dataset_type in ["train", "val", "test"]:
        _list = []
        for label in range(self.num_class):
          path = os.path.join(self.dataset_dir, str(label), dataset_type)
          _dataset = PatchDataset(path, label, num_classes=self.num_class, 
                                  one_hot=True)
          _list.append(_dataset)

        if "train" == dataset_type:
          self.train_dataset = ConcatDataset(_list)
        elif "val" == dataset_type:
          self.val_dataset = ConcatDataset(_list)
        elif "test" == dataset_type:
          self.test_dataset = ConcatDataset(_list)

      self.train_count = len(self.train_dataset)
      self.val_count = len(self.val_dataset)
      self.test_count = len(self.test_dataset)

    print(f"Number of samples used for training: {self.train_count}")
    print(f"Number of samples used for validation: {self.val_count}")
    print(f"Number of samples used for test: {self.test_count}")
  
  def create_data_loader(self, source):
    """
    Generic data loader function

    :param source: the type of dataset

    :return: Returns the constructed dataloader
    """
    datasets = {
      "train": self.train_dataset,
      "val": self.val_dataset,
      "test": self.test_dataset,
    }

    return DataLoader(datasets[source], batch_size=self.batch_size, 
                      shuffle=False, drop_last=False, 
                      num_workers=self.num_workers)
  
  def train_dataloader(self):
    """
    :return: output - Train data loader for the given input
    """
    return self.create_data_loader(source="train")

  def val_dataloader(self):
    """
    :return: output - Validation data loader for the given input
    """
    return self.create_data_loader(source="val")

  def test_dataloader(self):
    """
    :return: output - Test data loader for the given input
    """
    return self.create_data_loader(source="test")
    

class LitTDClassifier(L.LightningModule):
  
  def __init__(self, num_class, backbone='resnext50_32x4d', pretrained=True, 
               lr=8e-4, weight_decay=3e-5):
    super(LitTDClassifier, self).__init__()

    def _get_fcn(input_size, output_size, depth):
      module_list = []
      module_list += [nn.Linear(input_size, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(),
              nn.Dropout(0.2), ]
      for _ in range(depth-3):
        module_list += [nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2)]
      module_list += [nn.Linear(512, 256),
              nn.BatchNorm1d(256),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(256, output_size), ]
      return tuple(module_list)

    self.lr = lr
    self.weight_decay = weight_decay
    
    self.cnn, feature_size = get_model_by_name(backbone, pretrained)
    self.classifier = nn.Sequential(
      *(_get_fcn(feature_size, num_class, 10))
    )

    self.scheduler = None
    self.optimizer = None
    self.val_outputs = []
    self.test_outputs = []
    
  def forward(self, x):
    """
    :param x: input data

    :return: output - Type of news for the given news snippet
    """
    output = self.cnn(x)
    output = self.classifier(output)
    output = nn.functional.softmax(output, 1)
    return output
  
  def training_step(self, batch, batch_idx):
    """
    Training the data as batches and returns training loss on each batch

    :param train_batch Batch data
    :param batch_idx: Batch indices

    :return: output - Training loss
    """
    inputs, target = batch

    output = self.cnn(inputs)
    output = self.classifier(output)
    output = nn.functional.softmax(output, 1)
    loss = torch.nn.functional.mse_loss(output, target)
    self.log(
      "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, 
      logger=True, sync_dist=True
    )
    return {"loss": loss}  

  def configure_optimizers(self):
    self.optimizer = torch.optim.Adam(
      self.parameters(), lr=self.lr, weight_decay=self.weight_decay
    )
    self.scheduler = {
      "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
          self.optimizer,
          mode="min",
          factor=0.2,
          patience=2,
          min_lr=1e-6,
          verbose=True,
      ),
      "monitor": "val_loss",
    }
    
    return [self.optimizer], [self.scheduler]

  def validation_step(self, batch, batch_idx):
    inputs, target = batch
    output = self.cnn(inputs)
    output = self.classifier(output)
    output = nn.functional.softmax(output, 1)
    loss = torch.nn.functional.mse_loss(output, target)
    self.val_outputs.append(loss)      
    correct = (torch.argmax(output, -1) == torch.argmax(target, -1)).sum().item()

    self.log(
      "val_loss", loss, logger=True, sync_dist=True
    )
    self.log(
      "val_acc", correct/target.shape[0], logger=True, sync_dist=True
    )
    return {"val_step_loss": loss}
  
  def on_validation_epoch_end(self):
    """
    Computes average validation accuracy
    """
    avg_loss = torch.stack(self.val_outputs).mean()
    self.log("val_loss", avg_loss, sync_dist=True)
    self.val_outputs.clear()

  def test_step(self, batch, batch_idx):
    inputs, target = batch
    output = self.cnn(inputs)
    output = self.classifier(output)
    output = nn.functional.softmax(output, 1)
    loss = torch.nn.functional.mse_loss(output, target)
    self.test_outputs.append(loss)   
    correct = (torch.argmax(output, -1) == torch.argmax(target, -1)).sum().item()

    self.log(
      "test_loss", loss, logger=True, sync_dist=True
    )
    self.log(
      "test_acc", correct/target.shape[0], logger=True, sync_dist=True
    )
    test_acc = correct/target.shape[0]
    return {"test_acc": test_acc}

  def on_test_epoch_end(self):
    """
    Computes average test accuracy score
    """
    print(self.test_outputs)
    avg_test_acc = torch.stack(self.test_outputs).mean()
    self.log("avg_test_acc", avg_test_acc)
    self.test_outputs.clear()


class TDLightningCLI(LightningCLI):
  def add_arguments_to_parser(self, parser):
    parser.link_arguments("data.num_class", "model.num_class")

