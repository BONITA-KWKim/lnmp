import os
import shutil
import torch
import tqdm
import argparse
from torch.utils.data import DataLoader
from dataset import InferPatchDataset
from models.model import TDClassifier


NUM_CLASSES = 3
BATCH_SIZE = 1
BACKBONE = 'resnext50_32x4d'

PICK_THRESHOLD = .95
# PICK_THRESHOLD = .85
# PICK_THRESHOLD = .70

def pick_unlabelled_images(preds, target_root):
  filenames = preds.keys()

  # def _ambiguous_tile(scores):
  #   #if scores[2] < 0.1 and 0.6 < scores[1] and scores[1] <= 0.8:
  #   if scores[2] < 0.1 and scores[0] < 0.4:
  #     return True
  #   return False

  for filename in filenames:
    pred = preds[filename]
    label = pred[0]
    scores = pred[1]
    
    # if _ambiguous_tile(scores):
    #   shutil.copy(filename, os.path.join(target_root, "ambiguous"))

    if scores[label] > PICK_THRESHOLD:
      shutil.copy(filename, os.path.join(target_root, str(label)))


def infer(model, device, dataloader):
  model.eval()
  r = dict()
  with torch.no_grad():
    for images, image_filenames in tqdm.tqdm(dataloader):
      images = images.to(device)
      output = model(images)
      pred_labels = torch.argmax(output, 1)

      r[image_filenames[0]] = [pred_labels.item(), output.tolist()[0]]

  return r


def init_arg():
  parser = argparse.ArgumentParser(description='Inference ...')

  parser.add_argument('--verbose', default='info', choices=['info', 'debug'], 
                      type=str, help='The level of log')
  parser.add_argument('--backbone', default='resnext50_32x4d', type=str,
                      help='...')
  parser.add_argument('--n_class', default=2, type=int,
                      help='...')
  parser.add_argument('--model_dir', default='./results', type=str,
                      help='...')
  parser.add_argument('--dataset_dir', default='./dataset', type=str,
                      help='...')
  
  args = parser.parse_args()
  return args


def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Device : {device}')
  print(f'Current Device : {torch.cuda.current_device()}')
  setting = {}
  infer_paths = []

  # arguments
  args = init_arg()
  MODEL_DIR = args.model_dir
  BACKBONE = args.backbone
  NUM_CLASSES = args.n_class
  # load model
  model = TDClassifier(NUM_CLASSES, BACKBONE)
  model.load_state_dict(torch.load(MODEL_DIR)["state_dict"])
  model = model.to(device)

  INFER_ROOT = args.dataset_dir
  INFER_SLIDES = os.listdir(INFER_ROOT)
  for slide in INFER_SLIDES:
    infer_dir = os.path.join(INFER_ROOT, slide)
    print(f"currnet working directory: {infer_dir}")

    # dataset
    infer_dataset = InferPatchDataset(infer_dir)
  
    infer_dataloader = DataLoader(
      infer_dataset, shuffle=False, drop_last=False, batch_size=BATCH_SIZE,
      num_workers=1,
    )
  
    #target_root = os.path.join(f"./infer/ovarian-v0_1/6/preds_{PICK_THRESHOLD}", slide)
    target_root = os.path.join(f"/data/rnd1712/dataset/ovarian/raw/patch/pos_preds_{PICK_THRESHOLD}", slide)
    if os.path.exists(target_root) is False:
      os.makedirs(target_root)

    # mkdir classes dir
    for l in range(NUM_CLASSES):
      target_sub_dir = os.path.join(target_root, str(l))
      if os.path.exists(target_sub_dir) is False:
        os.makedirs(target_sub_dir)
    # mkdir ambiguous
    target_sub_dir = os.path.join(target_root, "ambiguous")
    if os.path.exists(target_sub_dir) is False:
      os.makedirs(target_sub_dir)
  
    # prediction
    preds = infer(model, device, infer_dataloader)
    # select proper images
    pick_unlabelled_images(preds, target_root)


"""Usage
python infer.py --verbose debug --n_class 3 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231030-1148/best.ckpt \
--dataset_dir /data/rnd1712/dataset/thyroid/test-infer 


[Ovarian]
# 231113 
nohup python infer.py --verbose debug --n_class 2 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231106-1648/best.ckpt \
--dataset_dir /data/rnd1712/dataset/ovarian/lnm/patched/0 &> infer-ovarian-data000-231113.log &

nohup python infer.py --verbose debug --n_class 2 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231106-1648/best.ckpt \
--dataset_dir /data/rnd1712/dataset/ovarian/lnm/patched/1 &> infer-ovarian-data001-231113.log &

nohup python infer.py --verbose debug --n_class 2 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231106-1648/best.ckpt \
--dataset_dir /data/rnd1712/dataset/ovarian/lnm/patched/2 &> infer-ovarian-data002-231114.log &
nohup python infer.py --verbose debug --n_class 2 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231106-1648/best.ckpt \
--dataset_dir /data/rnd1712/dataset/ovarian/lnm/patched/3 &> infer-ovarian-data003-231114.log &
nohup python infer.py --verbose debug --n_class 2 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231106-1648/best.ckpt \
--dataset_dir /data/rnd1712/dataset/ovarian/lnm/patched/4 &> infer-ovarian-data004-231114.log &
nohup python infer.py --verbose debug --n_class 2 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231106-1648/best.ckpt \
--dataset_dir /data/rnd1712/dataset/ovarian/lnm/patched/5 &> infer-ovarian-data005-231114.log &
nohup python infer.py --verbose debug --n_class 2 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231106-1648/best.ckpt \
--dataset_dir /data/rnd1712/dataset/ovarian/lnm/patched/6 &> infer-ovarian-data006-231114.log &


[v1_0_0] thyroid
python infer.py --verbose debug --n_class 3 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231030-1148/best.ckpt \
--dataset_dir /data/rnd1712/dataset/thyroid/slides/patch/patch-TC_04_01_filtered 

python infer.py --verbose debug --n_class 3 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231030-1148/best.ckpt \
--dataset_dir /data/rnd1712/dataset/thyroid/slides/patch/patch-TC_04_02_filtered 
"""

'''nohup python infer.py --verbose debug --n_class 3 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231030-1148/best.ckpt \
--dataset_dir /data/rnd1712/dataset/thyroid/slides/patch/patch-TC_04_01_filtered &> TC_04_01.log &

nohup python infer.py --verbose debug --n_class 3 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231030-1148/best.ckpt \
--dataset_dir /data/rnd1712/dataset/thyroid/slides/patch/patch-TC_04_02_filtered &> TC_04_02.log &

nohup python infer.py --verbose debug --n_class 3 \
--model_dir /data/rnd1712/lnmp/src/stage1/results/20231030-1148/best.ckpt \
--dataset_dir /data/rnd1712/dataset/thyroid/slides/patch/patch-TC_04_03_filtered &> TC_04_03.log &

'''


if __name__ == "__main__":
  main()
