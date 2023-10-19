import os
import tqdm
import torch
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset

from dataset import PrognosisDataset
from model import PrognosisClassifier


def get_dataloader(dataset_path, dataset_type, sub_path, num_classes, 
                   batch_size, num_workers):
  # dataset
  train_dataset_pos = PrognosisDataset(
    os.path.join(dataset_path, dataset_type[0], sub_path[1]), 
    1, num_classes=num_classes, one_hot=True)
  train_dataset_neg = PrognosisDataset(
    os.path.join(dataset_path, dataset_type[0], sub_path[0]), 
    0, num_classes=num_classes, one_hot=True)
  train_dataset = ConcatDataset([train_dataset_pos, train_dataset_neg])

  valid_dataset_pos = PrognosisDataset(
    os.path.join(dataset_path, dataset_type[1], sub_path[1]), 
    1, num_classes=num_classes, one_hot=True)
  valid_dataset_neg = PrognosisDataset(
    os.path.join(dataset_path, dataset_type[1], sub_path[0]), 
    0, num_classes=num_classes, one_hot=True)
  valid_dataset = ConcatDataset([valid_dataset_pos, valid_dataset_neg])

  # dataloader
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, drop_last=False, 
                                num_workers=num_workers)
  valid_dataloader = DataLoader(valid_dataset, batch_size=1, 
                                shuffle=False, drop_last=False, 
                                num_workers=num_workers)

  return train_dataloader, valid_dataloader


from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
def train_epoch(model, device, dataloader, loss_fn, optimizer):
  loss, correct = .0, 0
  model.train()

  y_true = []
  y_pred = []
  for images, labels in dataloader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    #t_image = train_transforms(images)
    probs = model(images)
    pred_labels = torch.argmax(probs, 1)

    loss = loss_fn(probs, labels)
    loss.backward()
    optimizer.step()
    labels = torch.argmax(labels, 1)

    correct += (pred_labels == labels).sum().item()
    loss += loss.item() / images.size(0)
    dataloader.set_postfix(
      acc=(pred_labels == labels).sum().item() / images.size(0) * 100,
      loss=loss.item() / images.size(0)
    )
    # assert ((pred_labels == labels).sum().item() / images.size(0) * 100) < 101, f'{pred_labels.size()}|{labels.size()}|{images.size()}'

    for item in pred_labels:
      y_pred.append(item.cpu())
    for item in labels:
      y_true.append(item.cpu())

  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)
  f1_macro = f1_score(y_true, y_pred, average='macro')
  f1_micro = f1_score(y_true, y_pred, average='micro')
  f1_weighted = f1_score(y_true, y_pred, average='weighted')
  print(f'[D] confusion matrix: {tn}, {fp}, {fn}, {tp}')
  print(f'[D] sensitivity: {sensitivity}. specificity: {specificity}')
  print(f'[D] f1 score: {f1_macro}, {f1_micro}, {f1_weighted}')

  return loss, correct


def valid_epoch(model, device, dataloader, num_classes, loss_fn):
  loss, correct = 0.0, 0
  model.eval()

  y_true = []
  y_pred = []

  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)

      probs = model(images)
      pred_labels = torch.argmax(probs, 1)
      loss = loss_fn(probs, labels)
      labels = torch.argmax(labels, 1)

      #print(f'D] pred: {pred_labels}\r\n     gt: {labels}')

      correct += (pred_labels == labels).sum().item()
      loss += loss.item() / images.size(0)
      dataloader.set_postfix(
          acc=(pred_labels == labels).sum().item() / images.size(0) * 100,
          loss=loss.item() / images.size(0))

      for item in pred_labels:
        y_pred.append(item.cpu())
      for item in labels:
        y_true.append(item.cpu())

  '''
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)
  f1_macro = f1_score(y_true, y_pred, average='macro')
  f1_micro = f1_score(y_true, y_pred, average='micro')
  f1_weighted = f1_score(y_true, y_pred, average='weighted')
  print(f'[D] confusion matrix: {tn}, {fp}, {fn}, {tp}')
  print(f'[D] sensitivity: {sensitivity}. specificity: {specificity}')
  print(f'[D] f1 score: {f1_macro}, {f1_micro}, {f1_weighted}')
  '''

  return loss, correct


def main():
  # init
  BATCH_SIZE = 8
  NUM_WORKERS = 2
  NUM_CLASSES = 2
  EPOCHS = 200
  #EPOCHS = 300
  #LR = 1e-4
  #WEIGHT_DECAY = 1e-5
  DEPTH=4
  LR = 3e-5
  WEIGHT_DECAY = 8e-6
  # gpu
  torch.autograd.set_detect_anomaly(True)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # dataset
  print('Dataloader')
  #dataset_path = '/data/rnd1712/dataset/breast/PROGNOSIS'
  #dataset_path = '/data/rnd1712/dataset/breast/PROGNOSIS'
  #dataset_path = '/data/rnd1712/dataset/breast/PROGNOSIS_v1_2'
  #dataset_path = '/data/rnd1712/dataset/breast/PROGNOSIS_1.4/x64'
  #dataset_path = '/data/rnd1712/dataset/breast/PROGNOSIS_1.4/x32'
  dataset_path = '/data/rnd1712/dataset/breast/PROGNOSIS_1.6'
  dataset_type = ['train', 'val']
  sub_path = ['neg', 'pos']
  train_dataloader, valid_dataloader = get_dataloader(dataset_path, 
                                         dataset_type, sub_path, NUM_CLASSES, 
                                         BATCH_SIZE, NUM_WORKERS)
  # model
  print('Model Create')
  loss_func = torch.nn.BCELoss()
  #loss_func = torch.nn.MSELoss()
  #model = PrognosisClassifier(NUM_CLASSES, 10)
  #model = PrognosisClassifier(NUM_CLASSES, 12)
  #model = PrognosisClassifier(NUM_CLASSES, 6)
  model = PrognosisClassifier(NUM_CLASSES, DEPTH)
  #model = PrognosisClassifier(NUM_CLASSES, 4)
  optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
  )

  # save
  now = datetime.now()
  save_path = './results/%s' % now.strftime('%Y%m%d-%H%M%S')
  if os.path.exists(save_path) is False:
    os.makedirs(save_path)

  TESTMODE = False
  #TESTMODE = True
  if TESTMODE:
    # load model
    #model_w_path = './results/20230808-145328/best.pth'
    #model_w_path = './results/20230809-104530/best.pth' # padding
    #model_w_path = './results/20230809-171029/best.pth' # padding 336
    #model_w_path = './results/20230809-162640/best.pth' # padding 512
    model_w_path = './results/20230809-162650/best.pth' # resize
    
    model.load_state_dict(torch.load(model_w_path))
    model = model.to(device)
    # dataloader
    test_dataset_pos = PrognosisDataset(
     #os.path.join(dataset_path, 'ev_test', 'pos'),
      os.path.join(dataset_path, 'val', 'pos'),
      1, num_classes=NUM_CLASSES, one_hot=True)
    test_dataset_neg = PrognosisDataset(
      os.path.join(dataset_path, 'val', 'neg'),
      #os.path.join(dataset_path, 'ev_test', 'neg'),
      0, num_classes=NUM_CLASSES, one_hot=True)
    test_dataset = ConcatDataset([test_dataset_pos, test_dataset_neg])
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, drop_last=False, 
                                num_workers=NUM_WORKERS)

    with tqdm.tqdm(test_dataloader, unit='batch') as vepoch:
      valid_loss, valid_correct = valid_epoch(
        model, device, vepoch, NUM_CLASSES, loss_func
      )
      valid_loss = valid_loss / len(valid_dataloader.sampler)
      valid_acc = valid_correct / len(valid_dataloader.sampler) * 100
  
      print(
        f'[EXTERNAL VAL] AVG VAL LOSS : {valid_loss:.4f} AVG VAL ACC : {valid_acc:.4f}'
      )
  else: 
    # save
    now = datetime.now()
    save_path = './results/%s' % now.strftime('%Y%m%d-%H%M%S')
    if os.path.exists(save_path) is False:
      os.makedirs(save_path)
    # model to gpu
    model = model.to(device)
    print('Start epochs')
    best_acc = 0.0
    best_loss = 10000.0
    for epoch in range(EPOCHS):
      with tqdm.tqdm(train_dataloader, unit='batch') as tepoch:
        tepoch.set_description(f'[Train EPOCH {epoch + 1}]')
        train_loss, train_correct = train_epoch(
          model, device, tepoch, loss_func, optimizer
        )
        train_loss = train_loss / len(train_dataloader.sampler)
        train_acc = train_correct / len(train_dataloader.sampler) * 100
        print(
          f'[Train EPOCH {epoch + 1}] AVG TRAIN LOSS : {train_loss:.4f} AVG TRAIN ACC : {train_acc:.4f}'
        )
  
      with tqdm.tqdm(valid_dataloader, unit='batch') as vepoch:
        vepoch.set_description(f'[VAL EPOCH {epoch + 1}]')
  
        valid_loss, valid_correct = valid_epoch(
          model, device, vepoch, NUM_CLASSES, loss_func
        )
        valid_loss = valid_loss / len(valid_dataloader.sampler)
        valid_acc = valid_correct / len(valid_dataloader.sampler) * 100
  
        print(
          f'[VAL EPOCH {epoch + 1}] AVG VAL LOSS : {valid_loss:.4f} AVG VAL ACC : {valid_acc:.4f}'
        )
  
      #best_acc = max(best_acc, valid_acc)
      if best_acc < valid_acc:
        best_acc = valid_acc
        best_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
        print(f"save path: {os.path.join(save_path,'best.pth')}")
  
    print(f'[BEST ACC]: {best_acc}')
    print(f'[BEST LOSS]: {best_loss}')


if __name__=="__main__":
  main()
