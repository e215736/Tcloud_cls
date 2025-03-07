
from __future__ import print_function, division

import glob
import os
import random
import copy
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
from pathlib import Path
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import torch.backends.cudnn as cudnn
from sam_mean_stdv0 import calculate_mean_std
import timm

# Training settings
epochs = 500
lr = 0.0001
seed = 42
split = 5
batch_size = 8

hflip_prob = 0.5
rotate_prob = 0.6
width_shift_range = 0.5
height_shift_range = 0.5

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

if torch.cuda.is_available():
    cudnn.benchmark = True

data_dirname = 'CCSN_3class'
data_path = f'./{data_dirname}'
use_mask = True
data_folder_name = 'train_sam'
test_folder_name = 'test_sam'

mean, std = calculate_mean_std(data_path, use_mask)

class TrainTransforms:
    def __init__(self, mean, std, hflip_prob=0.5, rotate_prob=0.5, 
                 width_shift_range=0.2, height_shift_range=0.2):
        self.mean = mean
        self.std = std
        self.hflip_prob = hflip_prob
        self.rotate_prob = rotate_prob
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

    def __call__(self, img):
        img = transforms.Resize((224, 224))(img)
        if random.random() < self.hflip_prob:
            img = transforms.RandomHorizontalFlip()(img)
        if random.random() < self.rotate_prob:
            angle = random.randint(-30, 30)
            img = transforms.functional.rotate(img, angle)
        if self.width_shift_range > 0 or self.height_shift_range > 0:
            width_shift = int(224 * random.uniform(-self.width_shift_range, self.width_shift_range))
            height_shift = int(224 * random.uniform(-self.height_shift_range, self.height_shift_range))
            img = transforms.functional.affine(
                img, angle=0, translate=(width_shift, height_shift), scale=1, shear=0
            )
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(self.mean, self.std)(img)
        return img

train_transforms = TrainTransforms(mean, std, hflip_prob, rotate_prob,
                                   width_shift_range, height_shift_range)

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_val_dataset = datasets.ImageFolder(os.path.join(data_path, data_folder_name), transform=None)
test_dataset = datasets.ImageFolder(os.path.join(data_path, test_folder_name), transform=None)
class_names = train_val_dataset.classes

class EfficientNetV2_B0(nn.Module):
    def __init__(self, class_num):
        super(EfficientNetV2_B0, self).__init__()
        self.base_model = timm.create_model('tf_efficientnetv2_b0', pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, class_num)
        )

    def forward(self, x):
        x = self.base_model(x)
        return x

def test_model(model, criterion, dataloaders, device, phase="test", fold=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    data_len = len(dataloaders[phase].dataset)

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    epoch_loss = running_loss / data_len
    epoch_acc = running_corrects.double() / data_len
    epoch_acc = epoch_acc.item() * 100.0

    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    conf_matrix = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})
    plt.title(f'EfficientNet Confusion Matrix ({phase}) Fold {fold + 1}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    save_dir = f"./image/conf_matrix/fold{fold + 1}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{phase}_conf_matrix_fold{fold + 1}.png")
    plt.close()

    return epoch_acc, epoch_loss

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=epochs, fold=None):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 50
    counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                counter = 0
                model_save_path = f"./model/CCSN_3class/aug-best_eff_model_fold{fold}.pth"
                torch.save(best_model_wts, model_save_path)
                print(f"Best model saved to: {model_save_path}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"EarlyStopping: No improvement for {patience} epochs. Stopping training.")
                    break

        if counter >= patience:
            break

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(best_model_wts)
    return model

kf = StratifiedKFold(n_splits=split, shuffle=True, random_state=seed)
targets = np.array(train_val_dataset.targets)

for fold, (train_index, val_index) in enumerate(kf.split(train_val_dataset, targets)):
    print(f"Fold {fold + 1}/{split}")

    train_subset = Subset(train_val_dataset, train_index)
    val_subset = Subset(train_val_dataset, val_index)

    dataloaders = {
        'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2,
                            collate_fn=lambda batch: (torch.stack([train_transforms(img) for img, _ in batch]),
                                                      torch.tensor([label for _, label in batch]))),
        'val': DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2,
                          collate_fn=lambda batch: (torch.stack([val_transforms(img) for img, _ in batch]),
                                                    torch.tensor([label for _, label in batch]))),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                           collate_fn=lambda batch: (torch.stack([val_transforms(img) for img, _ in batch]),
                                                     torch.tensor([label for _, label in batch])))
    }

    model = EfficientNetV2_B0(class_num=len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model = train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=epochs, fold=fold)
    test_model(model, criterion, dataloaders, device, phase="test", fold=fold)
