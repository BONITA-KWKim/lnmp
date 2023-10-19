import glob
import cv2 as cv
import torch
import torchvision
from torch.utils.data import Dataset


class BreastPatchDataset(Dataset):
  def __init__(self, path: str, label, num_classes, one_hot, transform=None):
    super(BreastPatchDataset, self).__init__()
    self.image_list = []
    self.target = label
    self.transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor()
    ])
    self.num_classes = num_classes
    self.one_hot = one_hot
    if transform is not None:
      self.transform = transform
    self.image_list = glob.glob(
      path+'/*.png', recursive=True) + glob.glob(path+'/*.jpg', recursive=True)

  def __getitem__(self, index):
    image = cv.imread(self.image_list[index])
    image = self.transform(image)
    if self.one_hot:
      return image, torch.nn.functional.one_hot(
        torch.tensor(self.target),
        self.num_classes).to(torch.float)
    else:
      return image, torch.tensor(self.target)

  def __len__(self):
    return len(self.image_list)


class NormalInferPatchDataset(Dataset):
  def __init__(self, path: str, transform=None):
    super(NormalInferPatchDataset, self).__init__()
    self.image_list = []
    self.transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor()
    ])
    if transform is not None:
      self.transform = transform
    self.image_list = glob.glob(
      path+'/*.png', recursive=True) + glob.glob(path+'/*.jpg', recursive=True)

  def __getitem__(self, index):
    image_filename = self.image_list[index]
    image = cv.imread(image_filename)
    image = self.transform(image)
    return image, image_filename

  def __len__(self):
    return len(self.image_list)


if __name__=="__main__":
  test_paths = ["/data/rnd1712/dataset/breast/lnm/breast-dataset-v1.0/100"
  "/data/rnd1712/dataset/breast/lnm/breast-dataset-v1.0/101"]
  dataset_list = []
  for i, path in enumerate(test_paths):
    dataset_test = BreastPatchDataset(
      path, i, num_classes=2, one_hot=False
    )
    dataset_list.append(dataset_test)
