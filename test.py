import os
from tqdm import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from dataloader import *
from model import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SensorFusion(device=device).to(device)

weights = r'multimodel\log\1C\models\weights_itr_95.pth'

if os.path.exists(weights):
    model.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')



if __name__ == '__main__':
    val_path = r'./val_data'

    filename_list = []
    for file in os.listdir(os.path.join(val_path, "D_real")):
        filename_list.append(file)

    datasets_val = DataLoader(MyDataset(val_path, filename_list), batch_size=1, shuffle=True)

    epoch_loss = 0
    model.eval()
    val_loop = tqdm.tqdm(datasets_val)
    for i, (image, us_image, F, P, D, label_F, label_P, label_D) in enumerate(val_loop):
        image = image.to(device)
        us_image = us_image.to(device)

        F = F.to(device)
        P = P.to(device)
        D = D.to(device)

        label_F = label_F.to(device)
        label_P = label_P.to(device)
        label_D = label_D.to(device)

        F_out, P_out, D_out = model(image, us_image, F, P, D)

        F_loss = loss_F(F_out, label_F)
        P_loss = loss_P(P_out, label_P)
        D_loss = loss_D(D_out, label_D)

        total_loss = (F_loss + P_loss + D_loss)
        epoch_loss += total_loss.item()
        if i == 0:
            tb.add_scalar("val/loss/F_loss", F_loss.item(), global_cnt_val)
            tb.add_scalar("val/loss/P_loss", P_loss.item(), global_cnt_val)
            tb.add_scalar("val/loss/D_loss", D_loss.item(), global_cnt_val)
            global_cnt_val += 1





