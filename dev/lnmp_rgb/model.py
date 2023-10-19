import torch
import torch.nn as nn
from torchsummary import summary

def get_model_by_name(modelname, pretrained):
    model = torch.hub.load(
      'pytorch/vision:v0.12.0', modelname, pretrained=pretrained)
    feature_size = 1000
    return model, feature_size


class PrognosisClassifier(nn.Module):
  def __init__(self, num_classes, depth):
    super(PrognosisClassifier, self).__init__()
    self.num_classes = num_classes
    self.depth = depth
    self.feature_size = 512
    '''
    self.cnn = nn.Sequential(
      # [x,3,512,512] -> [x,16,508,508]
      nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
      nn.ReLU(),
      # [x,16,508,508] -> [x,16,254,254]
      nn.MaxPool2d(kernel_size=2,stride=2), 
      # [x,16,254,254] -> [x,32,250,250]
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
      nn.ReLU(),
      # [x,32,250,250] -> [x,64,246,246]
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
      nn.ReLU(),
      # [x,64,246,246] -> [x,64,123,123]
      #nn.MaxPool2d(kernel_size=2,stride=2), 
      # [x,64,123,123] -> [x,128,119,119]
      #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
      #nn.ReLU(),
      # [x,32,250,250] -> [x,32,125,125]
      #nn.MaxPool2d(kernel_size=2,stride=2), 
      # [x,32,250,250] -> [x,32,125,125]
      nn.AdaptiveMaxPool2d((32,32)),
      nn.Flatten()
    )

    self.classifier = nn.Sequential(
      #nn.Linear(32*125*125, self.feature_size),
      nn.Linear(64*32*32, self.feature_size),
      #nn.Linear(16*32*32, self.feature_size),
      nn.BatchNorm1d(self.feature_size),
      nn.ReLU(),
      nn.Dropout(0.2), 
      *(self._get_fcn(self.feature_size, self.num_classes, self.depth))
    )
    '''
    #self.cnn, feature_size = get_model_by_name('vgg16', True)
    self.cnn, feature_size = get_model_by_name('resnet18', True)
    self.classifier = nn.Sequential(
      nn.Linear(feature_size, self.num_classes)
    )
    '''
    ''''''
    self.classifier = nn.Sequential(
      *(self._get_fcn(feature_size, self.num_classes, self.depth))
    )
    '''


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
    module_list += [nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, output_size), ]
    return module_list


  def forward(self, x):
    out = self.cnn(x)
    #out = out.view(-1, 32*125*125)
    #print(f'out: {out.shape}')
    out = self.classifier(out)
    out = nn.functional.softmax(out, 1)
    return out

if __name__=="__main__":
  model = PrognosisClassifier(2, 5)
  summary(model, (3, 512, 512), device='cpu')
