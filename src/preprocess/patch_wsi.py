import os
import sys
import math
import numpy as np
import random
import torch
import torchvision
import cv2

from time import time
# from absl import (app, flags, logging)
# from openslide import OpenSlide
# from PIL import Image, ImageDraw
from itertools import permutations
from tqdm import tqdm

from src.utils.wsi_map import WSIMap
# from models.bladder_classification import BCNet

import argparse
def init_arg():
  parser = argparse.ArgumentParser(description='WSI patching process.')

  parser.add_argument('--verbose', default='info', choices=['info', 'debug'], 
                      type=str, help='The level of log')
  parser.add_argument('--seg_level', default=0, type=int,
                      help='WSI magnification level')
  parser.add_argument('--w_size', default=32, type=int,
                      help='Overlapping window size')
  parser.add_argument('--mag_tile_size', default=8, type=int,
                      help='Magnification size for tile size. For example, if w_size were 32 and mag_tile_size were 8, tile size is 256*256(32*8)')
  parser.add_argument('--shrink_ratio', default=10, type=int,
                      help='Shrink ration for tissue detection')
  parser.add_argument('--input', default='./dataset', type=str,
                      help='Dataset directory')
  parser.add_argument('--output', default='./results', type=str,
                      help='Directory to save')
  
  args = parser.parse_args()
  return args

def prepare_each_image(data, transform, slice_type:str='vertical', 
                       slice_count:int=4, make_permutations:bool=False):
  assert transform is not None, 'Transform is None'
  #data = cv.cvtColor(data, cv.COLOR_BGR2HSV)
  size = data.shape[0] // slice_count
  assert data.shape[0] == data.shape[1], \
    'Image must have same width and height.: ' + str(data.shape)
  cropped_data_list = []

  if slice_count == 0:
      data = transform(data)
      return data

  if slice_type == 'both':
    for i in range(slice_count):
      row_data = []
      for j in range(slice_count):
        cropped_data = data[i*size:i *
                            size+size, j*size:j*size+size]
        cropped_data = transform(cropped_data)
        row_data.append(cropped_data)
      cropped_data_list.append(row_data)

  elif slice_type == 'vertical':
    for i in range(slice_count):
      cropped_data = data[0:data.shape[0], i*size:i*size+size]
      cropped_data = transform(cropped_data)
      cropped_data_list.append(cropped_data)

  elif slice_type == 'horizontal':
    for i in range(slice_count):
      cropped_data = data[i*size:i*size+size, 0:data.shape[1]]
      cropped_data = transform(cropped_data)
      cropped_data_list.append(cropped_data)
  else:
    logging.error('Wrong slice type!!')
    assert False

  if not make_permutations:
    if slice_type == 'both':
      temp = []
      for i in range(slice_count):
        temp += cropped_data_list[i]
      temp = torch.stack(temp)
      return temp
    else:
      return torch.stack(cropped_data_list)
  else:
    if slice_type == 'both':
      new_list = []
      for i in range(len(cropped_data_list)):
        cropped_data_list[i] = list(permutations(
          cropped_data_list[i], slice_count))
      for i in range(len(cropped_data_list[0])):
        data = []
        for j in range(slice_count):
          data += list(cropped_data_list[j][i])
        new_list.append(torch.stack(data))
      cropped_data_list = new_list
    else:
      cropped_data_list = list(permutations(cropped_data_list, slice_count))
    
    for data in cropped_data_list:
      if slice_type == 'both':
        tensor_data = data
      else:
        tensor_data = torch.stack(data)
      return tensor_data


def random_class():
  # sample_class = [0, 1, 2, 3, 4]
  sample_class = [0, 1, 2]
  return random.choice(sample_class)


def get_colour(no):

  def _color_to_np_color(color: str, transparent:int=100) -> np.ndarray:
    """
    Convert strings to NumPy colors.
    Args:
        color: The desired color as a string.
    Returns:
        The NumPy ndarray representation of the color.
    """
    colors = {
      "white": np.array([255, 255, 255, transparent]),
      "pink": np.array([255, 108, 180, transparent]),
      "black": np.array([0, 0, 0, transparent]),
      "red": np.array([255, 0, 0, transparent]),
      "purple": np.array([225, 225, 0, transparent]),
      "yellow": np.array([255, 255, 0, transparent]),
      "orange": np.array([255, 127, 80, transparent]),
      "blue": np.array([0, 0, 255, transparent]),
      "green": np.array([0, 255, 0, transparent])  }
    return colors[color]

  c = {
    0: _color_to_np_color('black'),
    1: _color_to_np_color('green'),
    2: _color_to_np_color('purple'),
    3: _color_to_np_color('blue'),
    4: _color_to_np_color('orange'),
  }
  return c[no]


def get_offset(openslide, window_size:int=32) -> tuple:
  # logging.debug(f'WSI properties\n{openslide.properties}')
  
  width = openslide.level_dimensions[0][0]
  height = openslide.level_dimensions[0][1]
  logging.debug(f'Level dimension: {openslide.level_dimensions[0]}')

  w_count = math.floor(width/window_size)
  h_count = math.floor(height/window_size)
  logging.debug(f'Count. W: {w_count}, Y: {h_count}')

  w_offset = width-(w_count*window_size)
  h_offset = height-(h_count*window_size)
  offset = (math.floor(w_offset/2), math.floor(h_offset/2)) # (w,h)
  logging.debug(f'Offset: {offset}')

  xx = []
  for i in range(w_count):
    xx.append(window_size*i+offset[0])
  logging.debug(f'xx: {len(xx)}. Start: {xx[0]}, Stop: {xx[-1]}')

  yy = []
  for i in range(h_count):
    yy.append(window_size*i+offset[1])
  logging.debug(f'yy: {len(yy)}. Start: {yy[0]}, Stop: {yy[-1]}')

  return width, height, xx, yy, w_count, h_count


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
  
  # logging.debug(f'Mean: {np.array(mean).mean()}, Std: {np.array(std).mean()}')
  return np.array(mean).mean(), np.array(std).mean()


import logging
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

  # 'application' code
  logger.debug('debug message')
  logger.info('info message')
  logger.warning('warn message')
  logger.error('error message')
  logger.critical('critical message')

  return logger


def main():
  # %% Init
  args = init_arg()
  OPENSLIDE_PATH = args.input
  SAVE_ROOT = args.output
  SEG_LEVEL = args.seg_level
  WINDOW_SIZE = args.w_size
  SHRINK_RATIO = args.shrink_ratio
  MAG_SIZE = args.mag_tile_size

  logger = get_logger(args.verbose)
  logger.info(f'Initialization\r\n\tTarget Image: {OPENSLIDE_PATH}\
    \r\n\tImage information. size: {(WINDOW_SIZE*MAG_SIZE)}, window: {WINDOW_SIZE}, window step: {MAG_SIZE}\
    \r\n\tTissue Detection Downsample ratio: {SHRINK_RATIO}')

  file_list = os.listdir(OPENSLIDE_PATH)
  filenames= [file for file in file_list if file.endswith(".svs")]
  logger.info(f"The length of images: {len(filenames)}")
  for filename in tqdm(filenames):
    # %% open WSI
    ''' Open WSI 
    '''
    t0 = time()
    wsimap = WSIMap(logger, os.path.join(OPENSLIDE_PATH, filename), 
                    speciman_type='TEST', window_size=WINDOW_SIZE)
    
    width, height = wsimap.get_size()
    xx, yy = wsimap.get_map_coordinate()
    w_count, h_count = wsimap.get_map_count()
    t1 = time()
    logger.info(f'Open WSI (elapsed: {round(t1-t0, 4)}s).')
   
    # %% Tissue Detection
    ''' Tissue Detection
    '''
    t0 = time()
    t_width = width//SHRINK_RATIO
    t_height = height//SHRINK_RATIO
    thumbnail_image = wsimap.get_thumbnail(t_width, t_height) # PIL image
    t1 = time()
    logger.info(f'Get thumbnail image (elapsed: {round(t1-t0, 4)}s). Type: {type(thumbnail_image)}')
  
    t0 = time()
    t_np = np.array(thumbnail_image.convert('RGB'))
    for i in range(w_count):
      for j in range(h_count):
    # for i in range(0, w_count, MAG_SIZE):
    #   for j in range(0, h_count, MAG_SIZE):
    #for i in range(0, w_count, MAG_SIZE*WINDOW_SIZE):
    #  for j in range(0, h_count, MAG_SIZE*WINDOW_SIZE):
        # numpy shape = (h, w, c)
        img_tile = t_np[yy[j]//SHRINK_RATIO:(yy[j]+(wsimap.window_size*MAG_SIZE))//SHRINK_RATIO,
                        xx[i]//SHRINK_RATIO:(xx[i]+(wsimap.window_size*MAG_SIZE))//SHRINK_RATIO, :]
        m, s = get_image_mean_and_std(img_tile)
        # tissue detection
        if m < 230 or s > 5.0:
          for ii in range(MAG_SIZE):
            if (i+ii) > (w_count-1): continue
            for jj in range(MAG_SIZE):
              if (j+jj) > (h_count-1): continue
              wsimap.info_map[i+ii][j+jj].active = True
    t1 = time()
    logger.info(f'Detect active area (elapsed: {round(t1-t0, 4)}s). window size: {wsimap.window_size}, magnification: {MAG_SIZE}')
    
    # %% get tiles and save it
    ''' tiles
    '''
    t0 = time()
    seglevel_image = wsimap.get_total_image_by_level(SEG_LEVEL) # PIL image
    seglevel_image = seglevel_image.convert('RGB') # PIL image is RGBA. Need to convert
    t1 = time()
    logging.info(f'Read total image (elapsed: {round(t1-t0, 4)}s). \
      Type: {type(seglevel_image)}. Memory size: {sys.getsizeof(seglevel_image)}')
    logging.debug(f'total image({seglevel_image.size}). \
      w: {seglevel_image.width}. h: {seglevel_image.height}.')
  
    t0 = time()
    seglevel_image_np = np.array(seglevel_image)
    t1 = time()
    logging.info(f'Level {SEG_LEVEL} total image (elapsed: {round(t1-t0, 4)}s). \
      Type: {type(seglevel_image_np)}. \
      Memory size: {sys.getsizeof(seglevel_image_np)}')
  
  
    t0 = time()
    if os.path.isdir(SAVE_ROOT) is False:
      os.makedirs(SAVE_ROOT)
  
    #for i in tqdm(range(0, w_count, MAG_SIZE//4)):
    #  for j in tqdm(range(0, h_count, MAG_SIZE//4), leave=False):
    img_name = os.path.splitext(filename)[0]
    save_dir = os.path.join(SAVE_ROOT, img_name)
    if os.path.isdir(save_dir) is False:
      os.makedirs(save_dir)

    for i in range(0, w_count, MAG_SIZE):
      for j in range(0, h_count, MAG_SIZE):
        # get region
        region = seglevel_image_np[yy[j]:yy[j]+(wsimap.window_size*MAG_SIZE),  
                                    xx[i]:xx[i]+(wsimap.window_size*MAG_SIZE), :]
        # inference
        if region.shape[0] != region.shape[1]: 
          logging.warning(f'Region. type: {type(region)}. shape: {region.shape}')
          continue
  
        # save results information
        active_cnt = 0
        for ii in range(MAG_SIZE):
          if (i+ii) > (w_count-1): continue
          for jj in range(MAG_SIZE):
            if (j+jj) > (h_count-1): continue
            if True == wsimap.info_map[i+ii][j+jj].active:
              active_cnt += 1
  
        logging.debug(f"active cnt: {active_cnt}")
        # save
        if active_cnt > 40:
          save_path = os.path.join(save_dir, f"{img_name}_{(i*WINDOW_SIZE)}_{j*WINDOW_SIZE}.png")
          cv2.imwrite(save_path, region)
  
    t1 = time()
    logging.info(f'save tiles (elapsed: {round(t1-t0, 4)}s)')
  

'''Usage
python patch_wsi.py --verbose debug --input /data/rnd1712/lnmp/test_dataset
'''
# %% main
if __name__=="__main__":
  main()
