import os
import random
import shutil

# 지정된 디렉토리의 파일 목록을 가져온다.
def get_file_list(dir_path):
    return os.listdir(dir_path)

# 파일 목록을 8:1:1로 나눈다.
def split_file_list(file_list):
    train_list = []
    val_list = []
    test_list = []

    # 파일 목록을 섞는다.
    random.shuffle(file_list)

    # 파일 목록을 8:1:1로 나눈다.
    train_list = file_list[:int(len(file_list) * 0.8)]
    val_list = file_list[int(len(file_list) * 0.8):int(len(file_list) * 0.9)]
    test_list = file_list[int(len(file_list) * 0.9):]

    return train_list, val_list, test_list

# 디렉토리 경로를 지정한다.
dir_root = "/data/rnd1712/dataset/thyroid/classification/v0-1"
dir_paths = []
for class_ in [0, 1, 2]:
  dir_paths.append(os.path.join(dir_root, str(class_)))
for dir_path in dir_paths:
  # 파일 목록을 가져온다.
  file_list = get_file_list(dir_path)

  # 파일 목록을 8:1:1로 나눈다.
  train_list, val_list, test_list = split_file_list(file_list)

  if os.path.exists(dir_path+"/train") is False:
     os.makedirs(dir_path+"/train")
  for img in train_list:
    shutil.copy(os.path.join(dir_path, img), os.path.join(dir_path+"/train", img))
  
  if os.path.exists(dir_path+"/val") is False:
     os.makedirs(dir_path+"/val")
  for img in val_list:
    shutil.copy(os.path.join(dir_path, img), os.path.join(dir_path+"/val", img))
  
  if os.path.exists(dir_path+"/test") is False:
     os.makedirs(dir_path+"/test")
  for img in test_list:
    shutil.copy(os.path.join(dir_path, img), os.path.join(dir_path+"/test", img))

  # 훈련 데이터, 검증 데이터, 테스트 데이터의 파일 목록을 출력한다.
  print("훈련 데이터:", train_list)
  print("검증 데이터:", val_list)
  print("테스트 데이터:", test_list)
