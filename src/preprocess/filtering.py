import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from src.utils.logger import get_logger


def init_arg():
  parser = argparse.ArgumentParser(description='WSI patching process.')

  parser.add_argument('--debug', default='info', choices=['info', 'debug'], 
                      type=str,
                      help='a boolean for log level')
  parser.add_argument('--input', default='./dataset', type=str,
                      help='Dataset directory')
  
  args = parser.parse_args()
  return args


def get_image_mean_and_std(img) -> tuple:
  N_CHANNELS = 3
  mean = [.0, .0, .0]
  std = [.0, .0, .0]
  for i in range(N_CHANNELS):
    if isinstance(img, np.ndarray):
      I = img
    else:
      I = np.asarray(img.convert('RGB'))
    mean[i] = round(I[:,:,i].mean(), 4)
    std[i] = round(I[:,:,i].std(), 4)
  return np.array(mean).mean(), np.array(std).mean()


def filtering(logger, image):
  filtered = False

  m, s = get_image_mean_and_std(image)
  logger.debug(f'mean: {round(m, 4)}, std: {round(s, 4)}')

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  maker_count = np.sum(np.array(gray) <= 20)
  logger.debug(f'pixel count: {maker_count}')

  if m > 230 and s < 20.0:
    filtered = True

  if maker_count > 10000:
    filtered = True

  b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
  c = np.count_nonzero(b)
  logger.debug(f"c count: {c}")
  ratio = (c/(gray.shape[0] * gray.shape[1]))
  logger.debug(f"ratio: {ratio}")
  if ratio < .32:
    filtered = True

  return filtered


def main():
  args = init_arg()
  logger = get_logger(args.debug)
  dataset = args.input
  logger.info('START')

  def _check_dir(root, new):
    created_dir = os.path.join(root, new)
    if os.path.isdir(created_dir) is False:
      os.makedirs(created_dir)
    return created_dir 


  def _filtering_images(files, save_root):
    for f in files:
      logger.debug(f'The image is loaded: {f}')
      # open image
      img = cv2.imread(os.path.join(root, f))
      r = filtering(logger, img)
      # save image
      if r is False:
        cv2.imwrite(os.path.join(save_root, f), img)


  logger.debug(f'dataset directory: {dataset}')
  filtered_dir = dataset+"_filtered"
  for idx, (root, dirs, files) in enumerate(tqdm(os.walk(dataset))):
    logger.debug(f'({idx}th)root: {root}')
    logger.debug(f'({idx}th)directory: {dirs}')
    logger.debug(f'({idx}th)files: {files}')

    if 0 == idx:
      _check_dir(filtered_dir, '')
      for dir_ in dirs:
        _check_dir(filtered_dir, dir_)
      save_root = filtered_dir
    else:
      _, wsi_id = os.path.split(root)
      logger.debug(f"wsi id: {wsi_id}")
      save_root = os.path.join(filtered_dir, wsi_id)
    _filtering_images(files, save_root)
        
  logger.info('END')

'''Usage
python filtering.py --input /data/rnd1712/lnmp/test_filtering_dataset
python filtering.py --input /data/rnd1712/dataset/thyroid/classification/rawdata

'''
if __name__=="__main__":
  main()
