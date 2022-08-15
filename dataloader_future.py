import os

import numpy as np
import torch
from torch.utils.data import Dataset
# from utils import *
from torchvision import transforms
from torch.nn.functional import one_hot
from PIL import Image
import cv2
import sys

transform = transforms.Compose([
    transforms.ToTensor()
])


def keep_image_size_open_rgb(path, size=(256, 256)):
    # img = Image.fromarray(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


class MyDataset(Dataset):
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.npy
        F_path = os.path.join(self.path, 'Force', segment_name)
        P_path = os.path.join(self.path, 'P_real', segment_name)
        D_path = os.path.join(self.path, 'D_real', segment_name)

        RGB_img_path = os.path.join(self.path, 'RGB_image', segment_name.replace("npy", "png"))
        US_img_path = os.path.join(self.path, 'cropped_ut_img', segment_name.replace("npy", "png"))

        label_D_20_path = os.path.join(self.path, 'label_D_20', segment_name)
        label_P_20_path = os.path.join(self.path, 'label_P_20', segment_name)
        label_F_20_path = os.path.join(self.path, 'label_F_20', segment_name)

        label_D_path = os.path.join(self.path, 'label_D', segment_name)
        label_P_path = os.path.join(self.path, 'label_P', segment_name)
        label_F_path = os.path.join(self.path, 'label_F', segment_name)

        F = np.load(F_path)
        P = np.load(P_path)
        D = np.load(D_path)

        label_D = np.load(label_D_path)
        label_P = np.load(label_P_path)
        label_F = np.load(label_F_path)

        label_D_20 = np.load(label_D_20_path)
        label_P_20 = np.load(label_P_20_path)
        label_F_20 = np.load(label_F_20_path)

        img = keep_image_size_open_rgb(RGB_img_path)
        us_img = keep_image_size_open_rgb(US_img_path)

        F = F * 100
        label_F = label_F * 100
        P = P * 10000
        label_P = label_P * 10000

        label_F_20 = label_F_20 * 100
        label_P_20 = label_P_20 * 10000

        return transform(img), transform(us_img), torch.Tensor(F), torch.Tensor(P), torch.Tensor(
            D), torch.Tensor(label_F), torch.Tensor(label_P), torch.Tensor(label_D), torch.Tensor(
            label_F_20), torch.Tensor(label_P_20), torch.Tensor(label_D_20)


if __name__ == "__main__":
    data_path = "./train_future"
    filename_list = []
    for file in os.listdir(os.path.join(data_path, "D_real")):
        filename_list.append(file)
    data = MyDataset(data_path, filename_list)
    print(data[0][9])
    print(data[20][6])

