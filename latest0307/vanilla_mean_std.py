"""
eval用ではない」なんでもない」
このPythonスクリプトmean_std.pyは、画像データセットの平均と標準偏差を計算するためのユーティリティスクリプトです。
これらの値は、画像データの正規化に使用されます。正規化は、モデルの学習を安定化させ、性能を向上させるために重要な前処理です。
"""
import os
import numpy as np
from PIL import Image
import torch
import torchvision

class CustomDataset(torch.utils.data.Dataset):
    """
    フォルダ内の画像ファイルを処理するためのカスタムデータセットクラス。
    torchvisionのtransformsを使用して画像を前処理する。
    """
    def __init__(self, root, transform=None):
        """
        初期化メソッド。データセットのルートディレクトリと変換を設定する。
        
        :param root: データセットのルートディレクトリ。
        :param transform: 画像に適用する変換（torchvision.transforms）。デフォルトはResizeとToTensor。
        """
        self.root = root
        self.transform = transform or torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        self.files = os.listdir(root)

    def __len__(self):
        """データセットのサイズ（画像の数）を返す。"""
        return len(self.files)

    def __getitem__(self, index):
        """
        指定されたインデックスの画像を読み込み、前処理を適用して返す。
        
        :param index: 取得する画像のインデックス。
        :return: 変換された画像データ。
        """
        file_path = os.path.join(self.root, self.files[index])
        try:
            image = Image.open(file_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except IOError:
            print(f"画像の読み込みに失敗しました: {file_path}")
            return None

def calculate_mean_std(dataset_path, use_mask):
    """
    指定されたデータセットのパスから、すべての画像データの平均と標準偏差を計算する。
    
    :param dataset_path: データセットのルートディレクトリ。
    :param use_mask: マスクされたデータセットを使用するかどうかのフラグ。
    :return: 画像データの平均と標準偏差のリスト。
    """
    #OIMG:test  SAM:test_mask
    labels = os.listdir(os.path.join(dataset_path, 'test'))
    #[cb,ns,other]
    mean_list = []
    std_list = []
    #=========================================change here if use mask======
    #      path = os.path.join(dataset_path, '????' if use_mask else 'train', label)
        #=================================================
    #OIMG:train  SAM:train_sam semseg:train_semseg
        #=================================================
    #=================================================
    for label in labels:
        path = os.path.join(dataset_path, 'train' if use_mask else 'train', label)
        dataset = CustomDataset(root=path)

        images = torch.stack([image for image in dataset if image is not None])
        std, mean = torch.std_mean(images, dim=(0, 2, 3))

        mean_list.append(mean.tolist())
        std_list.append(std.tolist())

    mean_list = np.mean(np.array(mean_list), axis=0)
    std_list = np.mean(np.array(std_list), axis=0)

    print('平均リスト:', mean_list)
    print('標準偏差リスト:', std_list)

    return mean_list, std_list
