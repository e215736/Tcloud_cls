"""
この train.py は、画像分類モデルを学習するための Python スクリプトです。
特に、k分割交差検証 と 転移学習 を用いて、モデルの汎化性能を高める工夫がされています。
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import EfficientNetB0
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchinfo import summary
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image
from mean_std_git import calculate_mean_std # 平均と標準偏差を計算するための独自関数
from torch.utils.data import Subset
from sklearn.model_selection import KFold

# バージョンとCUDAの可用性をチェック
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    cudnn.benchmark = True  # ベンチマークモードを有効化して、計算の高速化を図る

plt.ion()   # インタラクティブモードを有効化

# データセットの設定
use_mask =  True  # マスク処理の有無
data_dirname = 'CCSN_v7'  # データセットのディレクトリ名
data_path = f'./{data_dirname}'  # データセットのパス

# 平均と標準偏差の計算
mean, std = calculate_mean_std(data_path, use_mask)

# マスク処理に基づいて、入力フォルダ名と作成するモデル名を設定
if use_mask: 
    train_folder_name = 'train_mask'
    test_folder_name = 'test_mask'
    model_name = 'gsam_efficientnetb0_epo20_mask'
else:
    train_folder_name = 'train'
    test_folder_name = 'test'
    model_name = 'O-IMG_efficientnetb0__epo20'
print(train_folder_name)
# ハイパーパラメータの設定
batch_size = 32  # バッチサイズ
epochs = 50      # エポック数
split = 5        # 交差検証の分割数

# データ変換の定義
data_transforms = {
    train_folder_name: transforms.Compose([    
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    test_folder_name: transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
}

# データセットとデータローダーの作成
image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x])
                  for x in [train_folder_name, test_folder_name]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, num_workers=8)
               for x in [train_folder_name, test_folder_name]}

# データセットのサイズとクラス名を取得
dataset_sizes = {x: len(image_datasets[x]) for x in [train_folder_name, test_folder_name]}
class_names = image_datasets[test_folder_name].classes

# デバイスの設定（GPUが利用可能な場合はGPUを使用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

def test_model(model, criterion, dataloaders, device):
    """
    モデルのテスト性能を評価する関数。

    Args:
    model (torch.nn.Module): テストするモデル。
    criterion (torch.nn.modules.loss): 損失関数。
    dataloaders (dict): 'test'キーでテストデータローダーにアクセスできる辞書。
    device (torch.device): モデルを実行するデバイス（CPUまたはGPU）。

    Returns:
    tuple: テスト精度とテスト損失を含むタプル。
    """
    model.eval()  # モデルを評価モードに設定
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    # データローダーからバッチをイテレート
    for inputs, labels in dataloaders[test_folder_name]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():  # 勾配計算を無効化
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    test_loss = running_loss / len(dataloaders[test_folder_name].dataset)
    test_acc = running_corrects.double() / len(dataloaders[test_folder_name].dataset)
    test_acc = test_acc.item()
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    # 混同行列の計算と表示
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(conf_matrix)

    return test_acc, test_loss

def cross_train_model(model, criterion, optimizer, scheduler, num_epochs, train_loader, val_loader, k_fold, device, data_name, model_name, train_folder_name='train'):
    """
    交差検証でモデルを訓練する関数。

    Args:
    model (torch.nn.Module): 訓練するモデル。
    criterion (torch.nn.modules.loss): 損失関数。
    optimizer (torch.optim.Optimizer): オプティマイザ。
    scheduler (torch.optim.lr_scheduler): 学習率スケジューラ。
    num_epochs (int): エポック数。
    train_loader (torch.utils.data.DataLoader): 訓練データローダー。
    val_loader (torch.utils.data.DataLoader): 検証データローダー。
    k_fold (int): 現在の交差検証のフォールド番号。
    device (torch.device): モデルを実行するデバイス（CPUまたはGPU）。
    writer (torch.utils.tensorboard.writer.SummaryWriter): TensorBoard用のライター。
    data_name (str): データセット名。
    model_name (str): モデル名を保存する際のベース名。
    train_folder_name (str, optional): 訓練データフォルダの名前。デフォルトは 'train'。

    Returns:
    torch.nn.Module: 訓練されたモデル。
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 学習曲線用のリスト
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    # データセットのサイズを取得
    dataset_sizes = {
        train_folder_name: len(train_loader.dataset),
        'val': len(val_loader.dataset)
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 各エポックは訓練フェーズと検証フェーズを持つ
        for phase in [train_folder_name, 'val']:
            if phase == train_folder_name:
                model.train()  # モデルを訓練モードに設定
                loader = train_loader
            else:
                model.eval()   # モデルを評価モードに設定
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # データローダーからバッチをイテレート
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # 勾配をゼロに初期化

                # forward
                with torch.set_grad_enabled(phase == train_folder_name):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 訓練フェーズの場合は、バックワードとオプティマイズ
                    if phase == train_folder_name:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == train_folder_name:
                scheduler.step()  # 学習率スケジューラのステップ

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = epoch_acc.item()  #  ここを追加

            # 学習曲線用のリストに値を追加
            if phase == train_folder_name:
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)


            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 検証フェーズで最良のモデルを保存
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_path = f'./model/{data_name}'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)  # os.mkdir から os.makedirs に変更して、必要に応じて中間ディレクトリも作成
                torch.save(model.state_dict(), f'{model_path}/{model_name}_{k_fold + 1}.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    # 学習曲線のプロットと保存
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./sam_graph/loss_{k_fold + 1}.png')
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'./sam_graph/accuracy_{k_fold + 1}.png')
    plt.close()

    # 最良のモデルの重みをロード
    model.load_state_dict(best_model_wts)
    return model

all_test_acc = []
all_test_loss = []

# データセットの準備
dataset = image_datasets[train_folder_name]

# k分割交差検証のインデックスを取得
kf = KFold(n_splits=split, shuffle=True, random_state=1)
split_inds = list(kf.split(dataset))

# 各フォールドで学習と評価を繰り返す
for fold in range(split):
    # 学習用と評価用のインデックスを取得
    train_inds, valid_inds = split_inds[fold]
    print(f'Fold {fold + 1}')
    # 学習用と評価用のサブセットを作成
    train_subset = Subset(dataset, train_inds)
    valid_subset = Subset(dataset, valid_inds)

    # 学習用と評価用のデータローダーを作成
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # モデルの定義
    class_num = len(class_names)  # クラスの数に応じて変更してください
    model = EfficientNetB0(class_num)
    model = model.to(device)

    # 損失関数と最適化手法の定義
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model_ft = cross_train_model(model, criterion, optimizer, scheduler, num_epochs=epochs, train_loader=train_loader, val_loader=valid_loader, k_fold=fold, device=device, data_name=data_dirname, model_name=model_name)

    test_acc, test_loss = test_model(model_ft, criterion, dataloaders, device)
    all_test_acc.append(test_acc)
    all_test_loss.append(test_loss)

print("all_test_acc: ", all_test_acc, "all_test_loss: ", all_test_loss)

print(f'Cross Validation Loss: {np.mean(all_test_loss):.4f} ± {np.std(all_test_loss):.4f}')
print(f'Cross Validation Accuracy: {np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f}')
print(f'Cross Validation Top Accuracy: {np.max(all_test_acc):.4f}')
