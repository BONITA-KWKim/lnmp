import os
import glob
import cv2 as cv
import torch
import torchvision
import lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class PatchDataset(Dataset):
  def __init__(self, path: str, label, num_classes, one_hot, transform=None):
    super(PatchDataset, self).__init__()
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
        torch.tensor(self.target),self.num_classes
      ).to(torch.float)
    else:
      return image, torch.tensor(self.target)

  def __len__(self):
    return len(self.image_list)


class InferPatchDataset(Dataset):
  def __init__(self, path: str, transform=None):
    super(InferPatchDataset, self).__init__()
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
    self.train_count = None
    self.val_count = None
    self.test_count = None

  def setup(self, stage=None):
    """
    Split the data into train, test, validation data

    :param stage: Stage - training or testing
    """
    '''
    - Dataset Directory
    root/
      0/
        train/
        val/
        test/
      1/
        train/
        val/
        test/
      ...
    '''
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
 
  def create_data_loader(self, source):
    """
    Generic data loader function

    :param source: the type of dataset

    :return: Returns the constructed dataloader
    """
    if "train"==source:
      dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                              shuffle=True, drop_last=True, 
                              num_workers=self.num_workers)
    elif "val"==source:
      dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, 
                              shuffle=False, drop_last=False, 
                              num_workers=self.num_workers)
    elif "test"==source:
      dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, 
                              shuffle=False, drop_last=False, 
                              num_workers=self.num_workers)

    return dataloader
  
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
 