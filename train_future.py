import os
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from dataloader_future import *
from model_future import *
from torchvision.utils import save_image

from tensorboardX import SummaryWriter
import datetime
import time
import os

import logging
import sys
import yaml


def create_folder_structure(log_folder):
    """
    Creates the folder structure for logging. Subfolders can be added here
    """
    base_dir = log_folder
    sub_folders = ["runs", "models"]

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    for sf in sub_folders:
        if not os.path.exists(os.path.join(base_dir, sf)):
            os.mkdir(os.path.join(base_dir, sf))


log_name = "1C"
log_folder = os.path.join("./log_future", log_name)
tb_prefix = log_name

create_folder_structure(log_folder)

log_path = os.path.join(log_folder, "log.log")
print_logger = logging.getLogger()
print_logger.setLevel(getattr(logging, 'INFO', None))
handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
for h in handlers:
    h.setFormatter(formatter)
    print_logger.addHandler(h)
tb = SummaryWriter(os.path.join(log_folder, "runs", tb_prefix))


def logprint(self, *args):
    """
    Wrapper for print statement
    """
    print_logger.info(args)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = r'train_future'
val_path = r'val_future'
val_ratio = 1

filename_list = []
for file in os.listdir(os.path.join(data_path, "D_real")):
    filename_list.append(file)

val_filename_list = []
for file in os.listdir(os.path.join(val_path, "D_real")):
    val_filename_list.append(file)

global_cnt_val = 0

datasets_train = DataLoader(MyDataset(data_path, filename_list), batch_size=8, shuffle=True)
datasets_val = DataLoader(MyDataset(val_path, val_filename_list), batch_size=1, shuffle=False)

train_len_data = len(datasets_train)
val_len_data = len(datasets_val)

model = SensorFusion(device=device).to(device)

# 载入预训练权重
weights = r''

if os.path.exists(weights):
    model.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

opt = optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.0,
)

loss_P = nn.MSELoss()
loss_F = nn.MSELoss()
loss_D = nn.MSELoss()

loss_P_20 = nn.MSELoss()
loss_F_20 = nn.MSELoss()
loss_D_20 = nn.MSELoss()

epoch = 0
while epoch < 101:
    model.train()
    # 训练集
    train_loop = tqdm.tqdm(datasets_train)
    epoch_loss = 0
    for i, (image, us_image, F, P, D, label_F, label_P, label_D, label_F_20, label_P_20, label_D_20) in enumerate(
            train_loop):
        image = image.to(device)
        us_image = us_image.to(device)

        F = F.to(device)
        P = P.to(device)
        D = D.to(device)

        label_F = label_F.to(device)
        label_P = label_P.to(device)
        label_D = label_D.to(device)

        label_F_20 = label_F_20.to(device)
        label_P_20 = label_P_20.to(device)
        label_D_20 = label_D_20.to(device)

        F_out, P_out, D_out, F_20_out, P_20_out, D_20_out = model(image, us_image, F, P, D)

        F_loss = loss_F(F_out, label_F)
        P_loss = loss_P(P_out, label_P)
        D_loss = loss_D(D_out, label_D)

        F_20_loss = loss_F_20(F_20_out, label_F_20)
        P_20_loss = loss_P_20(P_20_out, label_P_20)
        D_20_loss = loss_D_20(D_20_out, label_D_20)

        total_loss = (F_loss + P_loss + D_loss + F_20_loss + P_20_loss + D_20_loss)

        train_loop.set_postfix(total_loss_=total_loss.item(), epoch_=epoch)
        epoch_loss += total_loss.item()
        opt.zero_grad()
        total_loss.backward()
        opt.step()
    print(epoch_loss)
    # 验证集
    if val_ratio != 0:
        epoch_loss = 0
        model.eval()
        val_loop = tqdm.tqdm(datasets_val)
        for i, (image, us_image, F, P, D, label_F, label_P, label_D, label_F_20, label_P_20, label_D_20) in enumerate(
                val_loop):
            image = image.to(device)
            us_image = us_image.to(device)

            F = F.to(device)
            P = P.to(device)
            D = D.to(device)

            label_F = label_F.to(device)
            label_P = label_P.to(device)
            label_D = label_D.to(device)

            label_F_20 = label_F_20.to(device)
            label_P_20 = label_P_20.to(device)
            label_D_20 = label_D_20.to(device)

            F_out, P_out, D_out, F_20_out, P_20_out, D_20_out = model(image, us_image, F, P, D)

            F_loss = loss_F(F_out, label_F)
            P_loss = loss_P(P_out, label_P)
            D_loss = loss_D(D_out, label_D)

            F_20_loss = loss_F_20(F_20_out, label_F_20)
            P_20_loss = loss_P_20(P_20_out, label_P_20)
            D_20_loss = loss_D_20(D_20_out, label_D_20)

            total_loss = (F_loss + P_loss + D_loss + F_20_loss + P_20_loss + D_20_loss)

            val_loop.set_postfix(total_loss_=total_loss.item(), epoch_=epoch)
            epoch_loss += total_loss.item()
            if i == 0:
                tb.add_scalar("val/loss/F_loss", F_loss.item(), global_cnt_val)
                tb.add_scalar("val/loss/P_loss", P_loss.item(), global_cnt_val)
                tb.add_scalar("val/loss/D_loss", D_loss.item(), global_cnt_val)
                global_cnt_val += 1
        print(epoch_loss)

    if epoch % 5 == 0:
        weight_path = os.path.join(log_folder, "models", "weights_itr_{}.pth".format(epoch))
        torch.save(model.state_dict(), weight_path)
        print('save successfully!')
    epoch += 1
