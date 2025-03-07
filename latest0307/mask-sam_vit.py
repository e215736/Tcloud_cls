from __future__ import print_function, division

import glob
import os
import random
import copy
import math

import matplotlib
matplotlib.use('Agg')  # Backend to prevent memory issues
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # CosineAnnealingLRをインポート
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import timm
from sklearn.model_selection import StratifiedKFold # StratifiedKFoldをインポート
from sklearn.metrics import confusion_matrix
import torch.backends.cudnn as cudnn
from mask_mean_std import calculate_mean_std


# Training settings
epochs = 500
lr = 0.0016  # 初期学習率を0.0016に変更
#gamma = 0.1
seed = 42
split = 5
batch_size = 8  # GPUメモリに合わせて調整

# データ拡張の確率
hflip_prob = 0.5
rotate_prob = 0.6
width_shift_range=0.5
height_shift_range=0.5

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

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

if torch.cuda.is_available():
    cudnn.benchmark = True

# データセットの設定
data_dirname = 'CCSN_3class'  # データセットのディレクトリ名
data_path = f'./{data_dirname}'  # データセットのパス
use_mask = True  # マスク処理の有無
data_folder_name = 'train_sam'
test_folder_name = 'test_sam'

# 平均と標準偏差の計算
mean, std = calculate_mean_std(data_path, use_mask)

# データ変換の定義 (オンザフライ方式、トレーニング用)
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
        # 1. Resize the image to a fixed size
        img = transforms.Resize((224, 224))(img)

        # 2. Random horizontal flip
        if random.random() < self.hflip_prob:
            img = transforms.RandomHorizontalFlip()(img)

        # 3. Random rotation
        if random.random() < self.rotate_prob:
            angle = random.randint(-30, 30)
            img = transforms.functional.rotate(img, angle)

        # 4. Random width and height shifts
        if self.width_shift_range > 0 or self.height_shift_range > 0:
            width_shift = int(224 * random.uniform(-self.width_shift_range, self.width_shift_range))
            height_shift = int(224 * random.uniform(-self.height_shift_range, self.height_shift_range))
            img = transforms.functional.affine(
                img, angle=0, translate=(width_shift, height_shift), scale=1, shear=0
            )

        # 5. Convert image to tensor and normalize
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(self.mean, self.std)(img)

        return img


# インスタンスを作成 (train用のみ)
train_transforms = TrainTransforms(mean, std, hflip_prob, rotate_prob,
                                   width_shift_range, height_shift_range)

# validation/test用の変換 (augmentationなし)
val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# データセットの作成
train_val_dataset = datasets.ImageFolder(os.path.join(data_path, data_folder_name), transform=None)
test_dataset = datasets.ImageFolder(os.path.join(data_path, test_folder_name), transform=None)  # transform=Noneに変更
class_names = train_val_dataset.classes


def test_model(model, criterion, dataloaders, device, phase="test"):
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

    # 混同行列
    conf_matrix = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})
    plt.title(f'sam-ViT Confusion Matrix ({phase}) Fold{fold+1}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.savefig(f"./image/sam-vit/conf_matrix/{phase}_confusion_matrix_fold{fold + 1}.png")
    plt.close()

    return epoch_acc, epoch_loss

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=epochs, fold=None):
    best_loss = float('inf')  # best_acc を best_loss に変更し、初期値を無限大に設定
    best_model_wts = copy.deepcopy(model.state_dict())  # 初期値
    best_epoch = 0
    patience = 50  # early stoppingのpatienceを設定
    counter = 0  # patienceカウンター

    # 学習曲線を格納するリスト
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


            if phase == 'train':
                train_losses.append(epoch_loss) # trainのlossを保存
                train_accs.append(epoch_acc.cpu().item()) # trainのaccuracyを保存
            else:
                val_losses.append(epoch_loss) # valのlossを保存
                val_accs.append(epoch_acc.cpu().item()) # valのaccuracyを保存



            if phase == 'val' : #and epoch_acc > best_acc:  変更点
                if epoch_loss < best_loss: # lossが減少したとき
                    best_loss = epoch_loss # best_lossを更新
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    counter = 0  # カウンターをリセット

                    # 各foldの最良モデルを保存
                    model_save_path = f"./model/CCSN_3class-sam-vit/aug-best_sam-vit_model_fold{fold}.pth" #保存パス
                    torch.save(best_model_wts, model_save_path)
                    print(f"    Best model saved to: {model_save_path}") # 保存場所を表示

                else:
                    counter += 1  # lossが減少してない場合、カウンターを増やす
                    if counter >= patience:  # カウンターがpatienceに達したらearly stopping
                        print(f"EarlyStopping: No improvement for {patience} epochs. Stopping training.")
                        break  # epochループを抜ける

        # epochループを抜ける条件を追加
        if counter >= patience:
            break



        if scheduler is not None:
            scheduler.step()
        print()

    print(f'Best val loss: {best_loss:.4f} at epoch {best_epoch+1}') #ベストloss,epochを表示 # 変更点

    # load best model weights
    model.load_state_dict(best_model_wts)

    # リストをbest_epochまでスライス
    train_losses = train_losses[:best_epoch+1]
    train_accs = train_accs[:best_epoch+1]
    val_losses = val_losses[:best_epoch+1]
    val_accs = val_accs[:best_epoch+1]


    return model, train_losses, train_accs, val_losses, val_accs

# k分割交差検証
kf = StratifiedKFold(n_splits=split, shuffle=True, random_state=seed)  # StratifiedKFoldに変更

all_val_accs = []
all_test_accs = []  # テスト精度のリスト
all_test_losses = []  # テスト損失のリスト
best_val_acc = 0.0  # 全フォールド中のベスト精度を追跡
best_model_wts = None

# 各foldのクラスごとのカウントを保存するリスト
val_counts_per_fold = []

# targets を numpy array に変換 (StratifiedKFoldで必要)
targets = np.array(train_val_dataset.targets)

for fold, (train_index, val_index) in enumerate(kf.split(train_val_dataset , targets)):
    print(f"Fold {fold + 1}/{split}")

    # データセットの分割
    train_subset = Subset(train_val_dataset, train_index)
    val_subset = Subset(train_val_dataset, val_index)

    # # validationデータのクラスごとの総数を出力 (変更点)
    # val_labels = [train_val_dataset.targets[i] for i in val_index]
    # val_counts = np.bincount(val_labels)
    # for i, count in enumerate(val_counts):
    #     print(f"Fold {fold+1} - Validation - {class_names[i]}: {count}")
    # データローダーの作成
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

    # モデルの定義
    model_name = f"vit_base_patch16_224_in21k_fold{fold+1}"
    model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=len(class_names))
    model.to(device)

    # 損失関数と最適化手法、スケジューラの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)  # AdamWを使用
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) # CosineAnnealingLRに変更


    # トレーニング
    best_fold_val_acc = 0.0  # このフォールド内でのベスト精度
    best_model_wts = copy.deepcopy(model.state_dict())  # ベストモデルの初期化

    # 学習 (返り値を受け取る)
    model, train_losses, train_accs, val_losses, val_accs = train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=epochs, fold=fold+1)


    # 学習曲線をプロット
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Fold {fold+1} - Loss')
    plt.savefig(f'./image/sam-vit/learning-curve/loss_fold{fold+1}.png')
    plt.close()


    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Fold {fold+1} - Accuracy')
    plt.savefig(f'./image/sam-vit/learning-curve/accuracy_fold{fold+1}.png')
    plt.close()



    # validation データでの評価
    val_acc, val_loss = test_model(model, criterion, dataloaders, device, phase="val")

    all_val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_fold = fold + 1
        best_model_wts = copy.deepcopy(model.state_dict())

    # 各foldのベストモデルでtestデータを評価
    print("-" * 20)
    print(f"Test dataset evaluation (Fold {fold+1}):")
    # testデータのdataloadersを作成 (各foldで作成)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                                collate_fn=lambda batch: (torch.stack([val_transforms(img) for img, _ in batch]),
                                                            torch.tensor([label for _, label in batch])))
    dataloaders['test'] = test_dataloader
    test_acc, test_loss = test_model(model, criterion, dataloaders, device, phase="test") # testデータの評価と混同行列の表示

    all_test_accs.append(test_acc) #test精度をリストに追加
    all_test_losses.append(test_loss)  #testロスをリストに追加

    # validationデータのクラスごとの総数をカウント
    val_labels = [targets[i] for i in val_index] # targetsからラベルを取得
    val_counts = np.bincount(val_labels)
    val_counts_per_fold.append(val_counts) # foldごとのカウントをリストに追加



# 結果出力部分をファイルに書き込む
res_file_path = "/home/endolab/TN_classification_cloud/image/sam-vit/res.txt"  # ファイルパス

with open(res_file_path, "w") as f:  # "w" で書き込みモードで開く
    print("-" * 20, file=f)  # file=f でファイルに書き込む
    print("Validation data counts per class per fold:", file=f)
    for fold, counts in enumerate(val_counts_per_fold):
        print(f"Fold {fold+1}:", file=f)
        for i, count in enumerate(counts):
            print(f"  {class_names[i]}: {count}", file=f)

    average_val_acc = np.mean(all_val_accs)

    print(f"Average Validation Accuracy: {average_val_acc:.4f}%", file=f)
    print(f"Best Fold: {best_fold}", file=f)

    print("all_test_accs: ", all_test_accs, "all_test_losses: ", all_test_losses, file=f)
    print(f'Average Test Loss: {np.mean(all_test_losses):.4f} ± {np.std(all_test_losses):.4f}', file=f)
    print(f'Average Test Accuracy: {np.mean(all_test_accs):.4f} ± {np.std(all_test_accs):.4f}%', file=f)
    print(f'Best Test Accuracy: {np.max(all_test_accs):.4f}%', file=f)