# -*- coding: utf-8 -*
import numpy as np
import cv2
import os
import json
import glob
import argparse
from openslide import OpenSlide

FILL_COLOR = (255, 255, 255)  # [Red, Green, Blue], the solid colors to fill irrelevant areas


import math
def get_wsi(wsi_path):
  try:
    openslide = OpenSlide(wsi_path)
  except:
    print(f'Could not open: {wsi_path}')

  mpp = openslide.properties['aperio.MPP']
  mpp = math.floor(float(mpp)*10)

  if mpp == 2:
    lv = 1
  else:
    # mpp == 5
    lv = 0
  
  width = openslide.level_dimensions[0][0]
  height = openslide.level_dimensions[0][1]
  wsi_img = openslide.read_region((0,0), 0, (width, height)).convert("RGB")
  wsi_img = np.asarray(wsi_img)

  #return wsi_img
  return lv, wsi_img


def get_annotation_points_and_bboxes_2(json_path):
  with open(json_path) as f:
    anno = json.load(f)

  annotation_points = []
  bboxes = []

  for idx, annotation in enumerate(anno):
    type_name = annotation["type"]
    if isinstance(type_name, int) is False: 
      #print(f"[W] type. {type(type_name)}, {type_name}")
      continue
    if int(type_name) == 3:
      annotation_points.append(annotation["polygon"])
      bboxes.append(get_bbox(annotation["polygon"]))

  return annotation_points, bboxes


def get_annotation_points_and_bboxes_new(json_path):
  with open(json_path) as f:
    anno = json.load(f)

  annotation_points = []
  bboxes = []

  for idx, annotation in enumerate(anno):
    try:
      type_name = annotation["properties"]["classification"]["name"]
    except:
      #print("[W] class name is invalid")
      continue
    
    #print("="*5, idx, "="*5)
    #print(f'[D] annotaion. name: {type_name}, type: {annotation["geometry"]["type"]}')
    # if type_name == "Tumor":
    if type_name == "Normal":
    # if type_name == "Invasive cancer_tubule" or type_name == "Invasive cancer_nontubule":
      if annotation['geometry']['type'] != "Polygon":
        #print(f"[W] ({idx}th) Invalid annotaion type: {annotation['geometry']['type']}")
        continue

      annotation_points.append(annotation["geometry"]["coordinates"][0])
      bboxes.append(get_bbox(annotation["geometry"]["coordinates"][0]))
    #print(f'[D] result. anno: {len(annotation_points)}, bboxes: {len(bboxes)}')
    #print("="*13)

  return annotation_points, bboxes


def get_annotation_points_and_bboxes(json_path):
    """get coordinate of each point in each annotated region, and get bounding box ([start_x, start_y, width, height]) of each annotated region"""
    with open(json_path) as f:
        asap_json = json.load(f)

    annotation_points = []
    bboxes = []

    # data only exist in 'positive'
    for i in asap_json['positive']:
        annotation_points.append(i['vertices'])
        bboxes.append(get_bbox(i['vertices']))

    return annotation_points, bboxes


def get_bbox(points):
    """get bounding box of an annotated region"""
    points = np.asarray(points, dtype=int)
    max_x_y = np.max(points, axis=0)
    min_x_y = np.min(points, axis=0)

    width_height = max_x_y - min_x_y
    bbox = min_x_y.tolist() + width_height.tolist()

    return bbox  # [start_x, start_y, width, height]


'''
rm -rf /data/118/MOHW/breast/dataset/breast_datatset/demo_cut_tumor_regions/; python cut_tumor_regions.py --wsi_dir_path '/data/118/MOHW/breast/dataset/breast_datatset/demo' \
--output_dir_path '/data/118/MOHW/breast/dataset/breast_datatset/demo_cut_tumor_regions'
'''
'''
rm -rf /data/118/MOHW/breast/dataset/breast_datatset/demo_v0.1_cut_tumor_regions;
python cut_tumor_regions.py --wsi_dir_path '/data/118/MOHW/breast/dataset/breast_datatset/demo_v0.1' \
--output_dir_path '/data/118/MOHW/breast/dataset/breast_datatset/demo_v0.1_cut_tumor_regions'

python cut_tumor_regions.py --wsi_dir_path '/data/118/MOHW/breast/dataset/breast-dataset-v1.0/converted-slides' \
--output_dir_path '/data/118/MOHW/breast/dataset/breast-dataset-v1.0/normal-tissue'

python cut_tumor_regions.py --wsi_dir_path '/data/118/MOHW/breast/dataset/breast-dataset-v1.0/converted-slides' \
--output_dir_path '/data/118/MOHW/breast/dataset/breast-dataset-v1.0/tumour-tissue'
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cut tumour regions of all WSIs')
    parser.add_argument('--wsi_dir_path', help='path of directory storing WSI and json', type=str, required=True)
    parser.add_argument('--output_dir_path', help='path of directory storing cut tumour regions', type=str, required=True)
    parser.add_argument('--not_filled_other_regions', help='not fill irrelevant areas with solid colors', action='store_false')
    args = parser.parse_args()

    assert not os.path.exists(args.output_dir_path), 'output_dir_path has existed, please change output_dir_path or remove it manually'

    # create output_dir_path
    os.makedirs(args.output_dir_path)
    #print('create directory: {}'.format(args.output_dir_path))

    # for wsi_path in glob.glob(os.path.join(args.wsi_dir_path, '*.jpg')):
    for wsi_path in glob.glob(os.path.join(args.wsi_dir_path, '*.svs')):
        wsi_id = os.path.splitext(os.path.basename(wsi_path))[0]
        json_path = wsi_path.replace('svs', 'json')

        # create directory to store tumour regions for each WSI
        os.makedirs(os.path.join(args.output_dir_path, wsi_id))
        print('create directory: {}'.format(os.path.join(args.output_dir_path, wsi_id)))

        # annotation_points, bboxes = get_annotation_points_and_bboxes(json_path)
        annotation_points, bboxes = get_annotation_points_and_bboxes_new(json_path)
        # annotation_points, bboxes = get_annotation_points_and_bboxes_2(json_path)

        if 0==len(annotation_points): continue

        lv, wsi_img = get_wsi(wsi_path)

        # scale down annotation points and bboxes

        for i, (ann_points, bbox) in enumerate(zip(annotation_points, bboxes)):
            tumour_save_path = os.path.join(args.output_dir_path, wsi_id, '{}_{}.jpg'.format(wsi_id, i))

            # extract a rectangular tumour region
            print(f'[D] bbox: {bbox}')
            x, y, width, height = bbox
            tumour_img = wsi_img[y: y + height, x: x + width]
            tumour_img = np.copy(wsi_img[y: y + height, x: x + width])
            # fill irrelevant areas with solid colors
            if args.not_filled_other_regions:
                mask_array = np.zeros((height, width), dtype=np.int32)

                ann_points = ann_points - np.asarray([x, y])  # compute the relative coordinate of each point in an annotated region
                # ann_points = np.unique(ann_points, axis=0)
                ann_points = np.expand_dims(ann_points, 0)
                ann_points = np.asarray(ann_points, dtype=np.int32)
                #print(f'[D] annotaion point: {np.sort(ann_points, axis=0)}')
                cv2.fillPoly(mask_array, ann_points, color=255)
                
                tumour_img[mask_array != 255] = FILL_COLOR  # fill irrelevant areas with solid colors
            #cv2.imwrite(tumour_save_path, tumour_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if lv == 1:
              # save down-scaled image
              scale_percent = 50 # percent of original size
              width = int(tumour_img.shape[1] * scale_percent / 100)
              height = int(tumour_img.shape[0] * scale_percent / 100)
              dim = (width, height)

              # resize image
              resized = cv2.resize(tumour_img, dim, interpolation = cv2.INTER_AREA)
              cv2.imwrite(tumour_save_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
              cv2.imwrite(tumour_save_path, tumour_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print('\t save {}'.format(tumour_save_path))
