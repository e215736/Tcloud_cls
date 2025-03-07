import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from timm import create_model
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime

# 平均と標準偏差を計算
from vanilla_mean_std import calculate_mean_std

mean, std = calculate_mean_std("./CCSN_3class/", False)

# データセットのパスと保存先フォルダ
data_dir = "./CCSN_3class/test"
categories = ["Cb", "Ns", "Other"]

# 保存先ディレクトリの絶対パス
base_dir = "/home/endolab/TN_classification_cloud/image/aug-vit-gridsearch"
acc_img_dir = os.path.join(base_dir, "acc_img")
miss_img_dir = os.path.join(base_dir, "miss_img")
acc_gradcam_dir = os.path.join(base_dir, "gradcam/acc")
miss_gradcam_dir = os.path.join(base_dir, "gradcam/miss")
conf_matrix_dir = os.path.join(base_dir, "conf_matrix")

# 保存先フォルダの準備
for dir_path in [acc_img_dir, miss_img_dir, acc_gradcam_dir, miss_gradcam_dir, conf_matrix_dir]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  # ディレクトリを削除して再作成
    os.makedirs(dir_path, exist_ok=True)

# CUDAデバイスの確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデルと重みのロード
model_path = "/home/endolab/TN_classification_cloud/model/CCSN_3class-vit/gridsearch/best_vit_large_patch16_224.augreg_in21k_batch16_AdamW_lr0.0016_fold4.pth"
model_name = "vit_large_patch16_224_in21k"
#model_name = "base_large_patch16_224_in21k"
model = create_model(model_name, pretrained=False, num_classes=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)  # モデルをCUDAデバイスに転送
model.eval()

# ViTの入力サイズに合わせたデータ変換
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# データセットの読み込み
dataset = ImageFolder(data_dir, transform=data_transforms)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Grad-CAM++を準備
target_layers = [model.blocks[-1].norm1]

# Grad-CAM用のカスタムフックを設定
def reshape_transform(tensor):
    result = tensor[:, 1:, :]
    height = width = int(result.size(1) ** 0.5)
    result = result.permute(0, 2, 1)
    result = result.view(result.size(0), result.size(1), height, width)
    return result

cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available(), reshape_transform=reshape_transform)

# 入力画像の事前処理用関数
def preprocess_for_gradcam(image_path):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = cv2.resize(img, (224, 224))
    return img

# Grad-CAM++結果の保存関数（ヒートマップを元画像に重ねる）
def save_gradcam_overlay(original_image_path, grayscale_cam, save_path):
    # 元画像の読み込み
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # OpenCVのBGRをRGBに変換
    original_image = cv2.resize(original_image, (224, 224))

    # Grad-CAMをカラーマップに変換
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # ヒートマップをRGBに変換

    # ヒートマップと元画像を重ねる
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    # 保存
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# カスタムターゲット関数
class TargetClass:
    def __init__(self, target_category):
        self.target_category = target_category

    def __call__(self, model_output):
        if model_output.dim() == 1:
            return model_output[self.target_category]
        elif model_output.dim() == 2:
            return model_output[:, self.target_category]
        else:
            raise ValueError(f"Unexpected model output dimensions: {model_output.dim()}")

# 全予測とラベルを保存するリスト
all_preds = []
all_labels = []

# 分類とGrad-CAM++の適用
for idx, (inputs, labels) in enumerate(data_loader):
    image_path, _ = dataset.samples[idx]
    img_name = os.path.basename(image_path)
    raw_img = preprocess_for_gradcam(image_path)
    inputs = inputs.to(device)  # 入力データをCUDAデバイスに転送
    labels = labels.to(device)  # ラベルもCUDAデバイスに転送

    # 推論
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    all_preds.append(preds.item())
    all_labels.append(labels.item())

    # 成功か失敗かの判定
    if preds == labels:
        img_dir = acc_img_dir
        gradcam_dir = acc_gradcam_dir
    else:
        img_dir = miss_img_dir
        gradcam_dir = miss_gradcam_dir

    # 元画像の保存
    copied_image_path = os.path.join(img_dir, img_name)
    shutil.copy(image_path, copied_image_path)

    # Grad-CAM++の適用
    target = TargetClass(preds.item())
    grayscale_cam = cam(input_tensor=inputs, targets=[target])

    if len(grayscale_cam) > 0:
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(raw_img, grayscale_cam, use_rgb=True)
        cam_image_uint8 = (cam_image * 255).astype(np.uint8)

        # Grad-CAM++結果を元画像に重ねて保存
        gradcam_path = os.path.join(gradcam_dir, f"gradcam_{img_name}")
        save_gradcam_overlay(copied_image_path, grayscale_cam, gradcam_path)


# 混同行列と分類レポートの生成
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, target_names=categories)

# 混同行列の表示と保存
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories, annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
conf_matrix_path = os.path.join(conf_matrix_dir, f"ViTorigconfusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(conf_matrix_path)
plt.close()

# 分類レポートの表示
print("Classification Report:")
print(class_report)

# 完了メッセージ
print("Grad-CAM++適用、混同行列および分類レポートの生成が完了しました！")
print(f"混同行列は {conf_matrix_path} に保存されました。")
