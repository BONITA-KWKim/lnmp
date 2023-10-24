import os
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader, ConcatDataset

# import lightning.pytorch as pl
import lightning as L
from lightning.pytorch.cli import LightningCLI

# from dataset import PatchDataset


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
    :param x -  input data

    :return: output - Type of news for the given news snippet
    """
    output = self.cnn(x)
    output = self.classifier(output)
    output = nn.functional.softmax(output, dim=1)
    return output
  
  def training_step(self, batch, batch_idx):
    """
    Training the data as batches and returns training loss on each batch

    :param batch - Batch data
    :param batch_idx - Batch indices

    :return: output - Training loss
    """
    inputs, target = batch

    output = self(inputs)
    loss = torch.nn.functional.mse_loss(output, target)
    self.log(
      "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, 
      logger=True, sync_dist=True
    )
    return {"loss": loss}  

  def configure_optimizers(self):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    self.scheduler = {
      "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
          self.optimizer,
          mode="min",
          factor=self.weight_decay,
          patience=10,
          min_lr=1e-6,
          verbose=True,
      ),
      "monitor": "val_loss",
    }
    
    return [self.optimizer], [self.scheduler]

  def validation_step(self, batch, batch_idx):
    inputs, target = batch

    output = self(inputs)
    loss = torch.nn.functional.mse_loss(output, target)
    self.val_outputs.append(loss)

    preds = torch.argmax(output, -1)
    gt = torch.argmax(target, -1)
    correct = (preds == gt).sum().item()

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

    output = self(inputs)
    loss = torch.nn.functional.mse_loss(output, target)

    preds = torch.argmax(output, -1)
    gt = torch.argmax(target, -1)
    correct = (preds == gt).sum().item()

    self.log(
      "test_loss", loss, logger=True, sync_dist=True
    )
    self.log(
      "test_acc", correct/target.shape[0], logger=True, sync_dist=True
    )
    test_acc = correct/target.shape[0]
    self.test_outputs.append((preds == gt).sum()/target.shape[0])   

    return {"test_acc": test_acc}

  def on_test_epoch_end(self):
    """
    Computes average test accuracy score
    """
    avg_test_acc = torch.stack(self.test_outputs).mean()
    self.log("avg_test_acc", avg_test_acc)
    self.test_outputs.clear()

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    inputs, _ = batch

    output = self(inputs)
    preds = torch.argmax(output, -1)

    return preds

class TDLightningCLI(LightningCLI):
  def add_arguments_to_parser(self, parser):
    parser.link_arguments("data.num_class", "model.num_class")

