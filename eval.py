"""
eval.py とmean_std.pyがセット。
この train.py は、画像分類モデルを学習するための Python スクリプトです。
特に、k分割交差検証 と 転移学習 を用いて、モデルの汎化性能を高める工夫がされています。
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from model import Cloudnet, Cloudmobinet, MobileNetV3, ResNet50, ResNet18, ResNet152, WideResNe101, EfficientNetB0
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import japanize_matplotlib
import shutil
import datetime
import cv2

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

from mean_std import calculate_mean_std

date = datetime.datetime.now()
date = str(date)

# GPUの設定
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データの前処理設定
data_dir = './CCSN_v7/'
mask = False  # マスク処理を行うかどうか

if mask:
    test_folder = 'test_mask'
    model_name = '/home/endolab/TN_classification_cloud/model/CCSN_v7/gsam_efficientnetb0_epo20_mask_5.pth'


else:
    test_folder = 'test'
    model_name = '/home/endolab/TN_classification_cloud/model/CCSN_v7/O-IMG_efficientnetb0__epo20_1.pth'

# データセットの平均と標準偏差を計算
mean, std = calculate_mean_std(data_dir, mask)

data_dir = os.path.join(data_dir, test_folder)
target = ["miss","acc"]



# データ変換処理の定義
data_transforms = transforms.Compose([
    transforms.Resize(224),  # EfficientNet B0の画像サイズに合わせてリサイズ
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # 計算した平均・標準偏差で正規化
])

# データセットとデータローダーの準備
batch_size = 1
image_datasets = datasets.ImageFolder(data_dir, data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, num_workers=8)
dataset_sizes = len(image_datasets)
class_names = image_datasets.classes

# バッチサイズ、データセットサイズ、デバイス情報の表示
print('バッチサイズ: ', batch_size)
print('テストデータセットサイズ: ', dataset_sizes)
print('使用デバイス: ', device)

date = datetime.datetime.now()
date = str(date)

#ファイルを空にする
acc_dir="/home/endolab/TN_classification_cloud/image/orig/acc_img"
miss_dir="/home/endolab/TN_classification_cloud/image/orig/miss_img"
shutil.rmtree(acc_dir)
shutil.rmtree(miss_dir)
os.makedirs(acc_dir)
os.makedirs(miss_dir)


# テスト関数の定義
def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    # テストデータの各バッチで予測値と正解ラベルを取得
    for index, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to(device)        
        labels = labels.to(device)        
        with torch.no_grad():
            outputs = model(inputs)            
            _, preds = torch.max(outputs, 1)

            # 正画像の保存
            if preds==labels:
                print(dataloaders.dataset.samples[index][0])                
                shutil.copy(dataloaders.dataset.samples[index][0], "./image/orig/acc_img/")
            # 誤画像の保存
            else:
                print(dataloaders.dataset.samples[index][0])                
                shutil.copy(dataloaders.dataset.samples[index][0], "./image/orig/miss_img")    

            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    test_loss = running_loss / dataset_sizes
    test_acc = running_corrects.double() / dataset_sizes

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    # 混同行列の計算
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(conf_matrix)

    # F値,recall,precisionの計算
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))



    # 混同行列の図の表示
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Cb","Ns","Other"], yticklabels=["Cb","Ns","Other"],annot_kws={"size": 20})
    plt.title('O-IMGConfusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig('./image/orig/conf_matrix/'+date+'_O-IMGconfusion_matrix.png')


class_num = len(class_names)


model = EfficientNetB0(class_num)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(model_name))

test_model(model, criterion)


# ここからGrad-CAMの処理
path_list = {}
# モデルの最終層を取得
target_layers = [model.base_model.features[-1]]
print(target_layers)
print("======================")
print(target)
# 画像の保存先を作成
# ここからGrad-CAMの処理
path_list = {}
# モデルの最終層を取得
target_layers = [model.base_model.features[-1]]
print("======================")
#print(target):['miss', 'acc']
# 画像の保存先を作成

# for i in target:
#     #print(os.listdir("./image/g-sam/miss_acc_img/"+i+"/"))
#     if not os.path.exists("./image/g-sam/gradcam/"+i):
#         os.mkdir("./image/g-sam/gradcam/"+i)
#     #classname:{Ns,Cb,Other}
#     for name in class_names:
#         print(name)
#         #print(os.listdir("./image/g-sam/miss_acc_img/"+i+"/"+name))
#         if not os.path.exists("./image/g-sam/gradcam/"+i+"/"+name):
#             os.mkdir("./image/g-sam/gradcam/"+i+"/"+name)
#         for filename in os.listdir("./image/g-sam/gradcam/"+i+"/"+name):
#             filepath = os.path.join("./image/g-sam/gradcam/"+i+"/"+name, filename)
#             print(filepath)
#             #print(filepath)
#             path_list.setdefault(i,[]).append(filepath)


# ... (その他のコードは同じ) ...

G_acc_dir="/home/endolab/TN_classification_cloud/image/orig/gradcam/acc"
G_miss_dir="/home/endolab/TN_classification_cloud/image/orig/gradcam/miss"
shutil.rmtree(G_acc_dir)
shutil.rmtree(G_miss_dir)
os.makedirs(G_acc_dir)
os.makedirs(G_miss_dir)

# 画像の読み込みとGrad-CAMの実行
for trg in target:
    # 読み込みディレクトリ
    input_dir = f"./image/orig/{trg}_img" 

    # 出力ディレクトリ
    output_dir = f"./image/orig/gradcam/{trg}"

    # サブディレクトリの名前のリスト
    sub_dirs = ["Cb", "Ns", "Other"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # サブディレクトリを作成
    for dir_name in sub_dirs:
        dir_path = os.path.join(output_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
    for filename in os.listdir(input_dir):
        # データ入力
        # 読み込みディレクトリ内のすべての画像ファイル名を順番に処理します。
        filepath = os.path.join(input_dir, filename)
        #BGR を RGB に変換します。
        rgb_img = cv2.imread(filepath, 1)[:, :, ::-1]
        #正規化
        rgb_img = np.float32(rgb_img) / 255
        #画像データをモデルに入力可能な形式に変換します。
        input_tensor = preprocess_image(rgb_img, mean=mean, std=std)

        # Note: input_tensor can be a batch tensor with several images!
        # Construct the CAM object once, and then re-use it on many images:

        """
        GradCAM オブジェクトを作成します。
        model: 訓練済みの画像分類モデル。
        target_layers: 重要度マップを計算する対象の層。ここでは、モデルの最終層（model.base_model.features[-1]）を指定しています。
        use_cuda: GPU を使用するかを指定します。True の場合は GPU を使用します。
        """
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        target_category = None


        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.

        #Grad-CAM を実行して、画像の各ピクセルの重要度を計算します。
        grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # ファイル名の先頭から - が出現するまでの文字列を取得
        file_prefix = filename.split("-")[0]

        # 保存場所の判断
        if file_prefix == "Ns":
            output_path = os.path.join(output_dir, "Ns", filename)
        elif file_prefix == "Cb":
            output_path = os.path.join(output_dir, "Cb", filename)
        else:
            output_path = os.path.join(output_dir, "Other", filename)

        # 可視化画像の保存
        #plt.imshow(visualization, vmin=0, vmax=255, interpolation='none')
        #plt.show()
        plt.imsave(output_path, visualization, vmin=0, vmax=255)