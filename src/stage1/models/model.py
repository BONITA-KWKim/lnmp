import torch
import torch.nn as nn
import lightning.pytorch as pl


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
      

class LitTDClassifier(pl.LightningModule):
  
  def __init__(self, num_classes, backbone='resnext50_32x4d', pretrained=True):
    super(LitTDClassifier, self).__init__()
    
    self.cnn, feature_size = get_model_by_name(backbone, pretrained)
    self.classifier = nn.Sequential(
      *(self._get_fcn(feature_size, num_classes, 10))
    )

    print(f"CNN. type: {type(self.cnn)}")
    
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
    return tuple(module_list)

  def training_step(self, batch, batch_idx):
    inputs, target = batch

    output = self.cnn(inputs)
    output = self.classifier(output)
    output = nn.functional.softmax(output, 1)
    loss = torch.nn.functional.mse_loss(output, target)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss   

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(
      self.parameters(), lr=8e-4, weight_decay=3e-5
    )
    return optimizer

  def validation_step(self, batch, batch_idx):
    inputs, target = batch
    output = self.cnn(inputs)
    output = self.classifier(output)
    output = nn.functional.softmax(output, 1)
    loss = torch.nn.functional.mse_loss(output, target)
          
    correct = (torch.argmax(output, -1) == torch.argmax(target, -1)).sum().item()

    self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
    self.log("val_acc", correct/target.shape[0], on_step=True, on_epoch=True, logger=True)


if __name__=="__main__":
  NUM_CLASSES = 2
  BACKBONE = "resnext50_32x4d"
  PRETRAINED = True
  model = TDClassifier(NUM_CLASSES, BACKBONE, PRETRAINED)
  l_model = LitTDClassifier(NUM_CLASSES, BACKBONE, PRETRAINED)

  # print(model)
  print(l_model)
