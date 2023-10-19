import glob
import cv2 as cv
import torch
import torchvision
from torch.utils.data import Dataset

RESIZE=128
#RESIZE=64
#RESIZE=224

#FLAG_RESIZE = True
FLAG_RESIZE = False
class PrognosisDataset(Dataset):
  def __init__(self, path: str, label, num_classes, one_hot, transform=None):
    super(PrognosisDataset, self).__init__()
    self.image_list = []
    self.target = label
    self.num_classes = num_classes
    self.one_hot = one_hot
    if transform is not None:
      self.transform = transform
    else:
      # transform
      self.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
      ])
    # read png or jpg image files
    self.image_list = (glob.glob(path+'/*.png', recursive=True) 
                      + glob.glob(path+'/*.jpg', recursive=True))

  def __getitem__(self, idx):
    im = cv.imread(self.image_list[idx])
    if FLAG_RESIZE:
      im = cv.resize(im, (RESIZE,RESIZE), interpolation = cv.INTER_AREA)
    else:
      # padding
      def _get_padding_size(l):
        #return (336-l)//2
        return (512-l)//2

      h, w, _ = im.shape
      top    = _get_padding_size(h)
      bottom = _get_padding_size(h)
      left   = _get_padding_size(w)
      right  = _get_padding_size(w)
      if h%2==1: top  += 1
      if w%2==1: left += 1
      im = cv.copyMakeBorder(im,top,bottom,left,right,cv.BORDER_CONSTANT,value=[0,0,0])

    im = self.transform(im)
    if self.one_hot:
      return im, torch.nn.functional.one_hot(
        torch.tensor(self.target),
        self.num_classes).to(torch.float)
    else:
      return im, torch.tensor(self.target)

  def __len__(self):
    return len(self.image_list)
