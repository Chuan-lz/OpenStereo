# @Time    : 2025年9月30日23点51分
# @Author  : liangzhen
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from stereo.datasets.dataset_utils.readpfm import readpfm
from .dataset_template import DatasetTemplate
from stereo.utils.common_utils import get_pos_fullres
from .dataset_template import DatasetTemplate, build_transform_by_cfg


class UAVDataset(DatasetTemplate):
      
    # """
    # 自定义双目数据集
    # 数据要求:
    #     left/   : 左图 png
    #     right/  : 右图 png
    #     disp/   : 左图视差 pfm
    # """
       
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)

        transform_config = data_cfg.DATA_TRANSFORM[mode.upper()]
        self.transform = build_transform_by_cfg(transform_config)
        
        self.split_file = ''
        self.data_list = []

        assert mode in ['training', 'evaluating', 'test'], f"Unsupported mode: {mode}"
        self.subdir = mode

        self.root = data_info.ROOT

        # 自动扫描数据文件
        left_dir = os.path.join(self.root, self.subdir, 'left')
        assert os.path.isdir(left_dir), f"Left image folder not found: {left_dir}"    

       # 自动生成数据列表，每项是去掉扩展名的文件名
        # 只取 .png 文件
        self.data_list = []
        for fname in sorted(os.listdir(left_dir)):
            if fname.lower().endswith('.png'):
                name = os.path.splitext(fname)[0]
                self.data_list.append([name])

        print(f"[UAVDataset] {self.subdir}: found {len(self.data_list)} samples")
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_id = self.data_list[idx][0]  # 如果是 [['000001'], ['000002']] 这种格式
        root = self.root  # training/val/test

        # 构造路径（根据 mode 切换子目录）
        left_img_path = os.path.join(root, self.subdir, 'left', f'{sample_id}.png')
        right_img_path = os.path.join(root, self.subdir, 'right', f'{sample_id}.png')
        disp_path = os.path.join(root, self.subdir, 'disp', f'{sample_id}.pfm')

        # 读取图像
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)

        # 读取视差图
        if os.path.exists(disp_path):
            disp = readpfm(disp_path)[0].astype(np.float32)
            disp = np.nan_to_num(disp, nan=0.0)
        else:
            h, w, _ = left_img.shape
            disp = np.zeros((h, w), dtype=np.float32)

        sample = {
            'left': left_img,     # [H, W, 3]
            'right': right_img,   # [H, W, 3]
            'disp': disp          # [H, W]
        }
        sample = self.transform(sample)
        
        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_img_path
        
        return sample


