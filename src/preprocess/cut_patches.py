# -*- coding: utf-8 -*
import cv2
import numpy as np
import os
import argparse


def padding(image_array, patch_size):
    """padding an image on the right and bottom with [255, 255, 255]"""
    height, width = image_array.shape[:2]

    right = int(np.ceil(width / patch_size) * patch_size - width)
    bottom = int(np.ceil(height / patch_size) * patch_size - height)

    # padding
    image_array = cv2.copyMakeBorder(image_array, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return image_array


def is_deprecated(image_array, blank_rate):
    """whether to deprecate the patches with blank ratio greater than max_blank_ratio"""
    blank_num = np.sum(image_array == (255, 255, 255)) / 3
    height, width = image_array.shape[:2]
    if blank_num / (height * width) >= blank_rate:
        return True
    else:
        return False

'''
python cut_patches.py --tumour_region_dir_path '/data/118/MOHW/breast/dataset/breast_datatset/demo_cut_tumor_regions' \
--size 256 \
--max_blank_ratio 0.3 \
--output_dir_path '/data/118/MOHW/breast/dataset/breast_datatset/demo_cut_patches' 

python cut_patches.py --tumour_region_dir_path '/data/118/MOHW/breast/dataset/breast_datatset/demo_v0.1_cut_tumor_regions' \
--size 256 \
--max_blank_ratio 0.2 \
--output_dir_path '/data/118/MOHW/breast/dataset/breast_datatset/demo_v0.1_cut_patches' 

python cut_patches.py --tumour_region_dir_path '/data/118/MOHW/breast/dataset/breast-dataset-v1.0/normal-tissue' \
--size 512 \
--max_blank_ratio 0.1 \
--output_dir_path '/data/118/MOHW/breast/dataset/breast-dataset-v1.0/normal-tissue-patches' 

python cut_patches.py --tumour_region_dir_path '/data/118/MOHW/breast/dataset/breast-dataset-v1.0/tumour-tissue' \
--size 512 \
--max_blank_ratio 0.1 \
--output_dir_path '/data/118/MOHW/breast/dataset/breast-dataset-v1.0/tumour-tissue-patches' 
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cut patches of all tumour regions')
    parser.add_argument('--tumour_region_dir_path', help='path of directory storing cut tumour regions (processed by cut_tumour_image.py)', type=str,
                        required=True)
    parser.add_argument('--size', help='size of cut patches', type=int, required=True)
    parser.add_argument('--max_blank_ratio', help='the max blank area ratio of a patch, the patch with ratio greater than this value will be filtered out',
                        type=float, required=True)
    parser.add_argument('--output_dir_path', help='path of directory storing cut patches', type=str, required=True)
    args = parser.parse_args()

    assert not os.path.exists(args.output_dir_path), 'output_dir_path has existed, please change output_dir_path or remove it manually'

    # create output_dir_path
    os.makedirs(args.output_dir_path)
    print('create directory: {}'.format(args.output_dir_path))

    for tumour_i in os.listdir(args.tumour_region_dir_path):

        os.makedirs(os.path.join(args.output_dir_path, tumour_i))
        print('create directory: {}'.format(os.path.join(args.output_dir_path, tumour_i)))

        for original_image in os.listdir(os.path.join(args.tumour_region_dir_path, tumour_i)):
            original_image_name = os.path.splitext(original_image)[0]
            original_image_path = os.path.join(args.tumour_region_dir_path, tumour_i, original_image)

            print('\tprocess {}'.format(original_image_path))
            try:
              big_image_array = cv2.imread(original_image_path)
            except:
              print(f'[W] Image is too big({original_image_path})')
              continue

            # padding a tumour image for cutting
            big_image_array = padding(big_image_array, args.size)

            # cut patches from a tumour image
            height, width = big_image_array.shape[:2]
            for y in np.arange(0, height, args.size):
                for x in np.arange(0, width, args.size):
                    small_image_array = big_image_array[y:y + args.size, x:x + args.size]
                    small_image_path = os.path.join(args.output_dir_path, tumour_i, '{}_{}_{}.jpg'.format(original_image_name, x, y))

                    # deprecate the patches with blank ratio greater than max_blank_ratio
                    if is_deprecated(small_image_array, args.max_blank_ratio):
                        print('\tdeprecated {}'.format(small_image_path))
                    else:
                        cv2.imwrite(small_image_path, small_image_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        print('\tsave {}'.format(small_image_path))
