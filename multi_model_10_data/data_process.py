import os
import shutil
import numpy

data_path = r"C:\Users\Paradox\Desktop\git\multimodel\multi_model_10_data"
aim_path = r"C:\Users\Paradox\Desktop\git\multimodel\val_data"

if not os.path.exists(os.path.join(aim_path, r'label_P')):
    os.makedirs(os.path.join(aim_path, r'label_P'))
if not os.path.exists(os.path.join(aim_path, r'label_F')):
    os.makedirs(os.path.join(aim_path, r'label_F'))
if not os.path.exists(os.path.join(aim_path, r'label_D')):
    os.makedirs(os.path.join(aim_path, r'label_D'))

if not os.path.exists(os.path.join(aim_path, r'D_real')):
    os.makedirs(os.path.join(aim_path, r'D_real'))
if not os.path.exists(os.path.join(aim_path, r'P_real')):
    os.makedirs(os.path.join(aim_path, r'P_real'))
if not os.path.exists(os.path.join(aim_path, r'Force')):
    os.makedirs(os.path.join(aim_path, r'Force'))

if not os.path.exists(os.path.join(aim_path, r'RGB_image')):
    os.makedirs(os.path.join(aim_path, r'RGB_image'))
if not os.path.exists(os.path.join(aim_path, r'cropped_ut_img')):
    os.makedirs(os.path.join(aim_path, r'cropped_ut_img'))

count = 0
for i in range(10, 11):
    file_path = os.path.join(data_path, "test{}".format(i))

    files = os.listdir(os.path.join(file_path, r'D_real'))
    for j in range(0, len(files)):
        shutil.copyfile(os.path.join(file_path, r'D_real', '{}.npy'.format(j + 1)),
                        os.path.join(aim_path, r'D_real', '{}.npy'.format(count)))
        shutil.copyfile(os.path.join(file_path, r'P_real', '{}.npy'.format(j + 1)),
                        os.path.join(aim_path, r'P_real', '{}.npy'.format(count)))
        shutil.copyfile(os.path.join(file_path, r'Force', '{}.npy'.format(j + 1)),
                        os.path.join(aim_path, r'Force', '{}.npy'.format(count)))

        shutil.copyfile(os.path.join(file_path, r'label_D', '{}.npy'.format(j + 1)),
                        os.path.join(aim_path, r'label_D', '{}.npy'.format(count)))
        shutil.copyfile(os.path.join(file_path, r'label_P', '{}.npy'.format(j + 1)),
                        os.path.join(aim_path, r'label_P', '{}.npy'.format(count)))
        shutil.copyfile(os.path.join(file_path, r'label_F', '{}.npy'.format(j + 1)),
                        os.path.join(aim_path, r'label_F', '{}.npy'.format(count)))

        shutil.copyfile(os.path.join(file_path, r'cropped_ut_img', '{}.png'.format(j + 1)),
                        os.path.join(aim_path, r'cropped_ut_img', '{}.png'.format(count)))
        shutil.copyfile(os.path.join(file_path, r'{}'.format(i), '{}.png'.format(j + 1)),
                        os.path.join(aim_path, r'RGB_image', '{}.png'.format(count)))
        count = count + 1
