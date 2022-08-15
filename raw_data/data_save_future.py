import os
import shutil
import numpy
import cv2
import numpy as np
import csv

aim_path = r"processed_data_future"
raw_path = r"multi_modal_02_08"
data_len = len(os.listdir(os.path.join(raw_path, "RGB_Image")))

if not os.path.exists(aim_path):
    os.makedirs(aim_path)
if not os.path.exists("train_future"):
    os.makedirs("train_future")
if not os.path.exists("val_future"):
    os.makedirs("val_future")

for i in range(1, data_len + 1):
    if not os.path.exists(os.path.join(aim_path, "test{}".format(i))):
        os.makedirs(os.path.join(aim_path, "test{}".format(i)))
    aim_path_n = os.path.join(aim_path, "test{}".format(i))

    if not os.path.exists(os.path.join(aim_path_n, "{}".format(i))):
        os.makedirs(os.path.join(aim_path_n, "{}".format(i)))
    if not os.path.exists(os.path.join(aim_path_n, r'cropped_ut_img')):
        os.makedirs(os.path.join(aim_path_n, r'cropped_ut_img'))

    if not os.path.exists(os.path.join(aim_path_n, r'D_real')):
        os.makedirs(os.path.join(aim_path_n, r'D_real'))
    if not os.path.exists(os.path.join(aim_path_n, r'Force')):
        os.makedirs(os.path.join(aim_path_n, r'Force'))
    if not os.path.exists(os.path.join(aim_path_n, r'P_real')):
        os.makedirs(os.path.join(aim_path_n, r'P_real'))

    if not os.path.exists(os.path.join(aim_path_n, r'D_expect')):
        os.makedirs(os.path.join(aim_path_n, r'D_expect'))
    if not os.path.exists(os.path.join(aim_path_n, r'P_expect')):
        os.makedirs(os.path.join(aim_path_n, r'P_expect'))

    files = os.listdir(os.path.join(raw_path, "RGB_Image", "{}".format(i)))
    # 读入csv
    csv_reader = csv.reader(open(os.path.join(raw_path, r"robot_state", "{}.csv".format(i))))
    lines = []
    for line in csv_reader:
        lines.append(line)

    for j in range(len(files)):
        # RGB
        shutil.copyfile(os.path.join(raw_path, r"RGB_Image", "{}".format(i), '{}.png'.format(j + 1)),
                        os.path.join(aim_path_n, "{}".format(i), '{}.png'.format(j + 1)))
        # US_image
        US_img = cv2.imread(os.path.join(raw_path, r"Ultrasound_image", "{}".format(i), '{}.png'.format(j + 1)))
        cropped_US_img = US_img[100:400, 140:500]
        cv2.imwrite(os.path.join(aim_path_n, r"cropped_ut_img", '{}.png'.format(j + 1)), cropped_US_img)

        # csv数据
        # P_expect 1,2,3
        pos_expect = [float(lines[-(j + 1)][0]), float(lines[-(j + 1)][1]), float(lines[-(j + 1)][2])]
        np.save(os.path.join(aim_path_n, r'P_expect', '{}.npy'.format(len(files) - j)), np.array(pos_expect))
        # P_real
        pos_real = [float(lines[-(j + 1)][3]), float(lines[-(j + 1)][4]), float(lines[-(j + 1)][5])]
        np.save(os.path.join(aim_path_n, r'P_real', '{}.npy'.format(len(files) - j)), np.array(pos_real))
        # D_expect
        D4_expect = [float(lines[-(j + 1)][6]), float(lines[-(j + 1)][7]), float(lines[-(j + 1)][8]),
                     float(lines[-(j + 1)][9])]
        np.save(os.path.join(aim_path_n, r'D_expect', '{}.npy'.format(len(files) - j)), np.array(D4_expect))
        # D_real
        D4_real = [float(lines[-(j + 1)][10]), float(lines[-(j + 1)][11]), float(lines[-(j + 1)][12]),
                   float(lines[-(j + 1)][13])]
        np.save(os.path.join(aim_path_n, r'D_real', '{}.npy'.format(len(files) - j)), np.array(D4_real))

        # Force
        Force = [float(lines[-(j + 1)][14]), float(lines[-(j + 1)][15]), float(lines[-(j + 1)][16])]
        np.save(os.path.join(aim_path_n, r'Force', '{}.npy'.format(len(files) - j)), np.array(Force))

path = r"processed_data_future"
data_len = len(os.listdir(path))
for i in range(1, data_len + 1):
    file_path = os.path.join(path, "test{}".format(i))
    if not os.path.exists(os.path.join(file_path, r'label_D')):
        os.makedirs(os.path.join(file_path, r'label_D'))
    if not os.path.exists(os.path.join(file_path, r'label_P')):
        os.makedirs(os.path.join(file_path, r'label_P'))
    if not os.path.exists(os.path.join(file_path, r'label_F')):
        os.makedirs(os.path.join(file_path, r'label_F'))

    if not os.path.exists(os.path.join(file_path, r'label_D_20')):
        os.makedirs(os.path.join(file_path, r'label_D_20'))
    if not os.path.exists(os.path.join(file_path, r'label_P_20')):
        os.makedirs(os.path.join(file_path, r'label_P_20'))
    if not os.path.exists(os.path.join(file_path, r'label_F_20')):
        os.makedirs(os.path.join(file_path, r'label_F_20'))

    files = os.listdir(os.path.join(file_path, r'D_real'))

    if os.path.exists(os.path.join(file_path, r'{}'.format(i), '.DS_Store')):
        os.remove(os.path.join(file_path, r'{}'.format(i), '.DS_Store'))
    if os.path.exists(os.path.join(file_path, r'cropped_ut_img', '.DS_Store')):
        os.remove(os.path.join(file_path, r'cropped_ut_img', '.DS_Store'))

    for j in range(len(files)):
        shutil.copyfile(os.path.join(file_path, r'D_expect', '{}.npy'.format(j + 1)),
                        os.path.join(file_path, r'label_D', '{}.npy'.format(j + 1)))
        shutil.copyfile(os.path.join(file_path, r'P_expect', '{}.npy'.format(j + 1)),
                        os.path.join(file_path, r'label_P', '{}.npy'.format(j + 1)))
        shutil.copyfile(os.path.join(file_path, r'Force', '{}.npy'.format(j + 1)),
                        os.path.join(file_path, r'label_F', '{}.npy'.format(j)))
        if j + 21 < len(files):
            shutil.copyfile(os.path.join(file_path, r'D_expect', '{}.npy'.format(j + 21)),
                            os.path.join(file_path, r'label_D_20', '{}.npy'.format(j + 1)))
            shutil.copyfile(os.path.join(file_path, r'P_expect', '{}.npy'.format(j + 21)),
                            os.path.join(file_path, r'label_P_20', '{}.npy'.format(j + 1)))
            shutil.copyfile(os.path.join(file_path, r'Force', '{}.npy'.format(j + 21)),
                            os.path.join(file_path, r'label_F_20', '{}.npy'.format(j)))
        else:
            shutil.copyfile(os.path.join(file_path, r'D_expect', '{}.npy'.format(len(files))),
                            os.path.join(file_path, r'label_D_20', '{}.npy'.format(j + 1)))
            shutil.copyfile(os.path.join(file_path, r'P_expect', '{}.npy'.format(len(files))),
                            os.path.join(file_path, r'label_P_20', '{}.npy'.format(j + 1)))
            shutil.copyfile(os.path.join(file_path, r'Force', '{}.npy'.format(len(files))),
                            os.path.join(file_path, r'label_F_20', '{}.npy'.format(j)))

    os.remove(os.path.join(file_path, r'label_D', '{}.npy'.format(len(files))))
    os.remove(os.path.join(file_path, r'label_P', '{}.npy'.format(len(files))))
    os.remove(os.path.join(file_path, r'label_F', '0.npy'))
    os.remove(os.path.join(file_path, r'label_D_20', '{}.npy'.format(len(files))))
    os.remove(os.path.join(file_path, r'label_P_20', '{}.npy'.format(len(files))))
    os.remove(os.path.join(file_path, r'label_F_20', '0.npy'))
    os.remove(os.path.join(file_path, r'{}'.format(i), '{}.png'.format(len(files))))
    os.remove(os.path.join(file_path, r'cropped_ut_img', '{}.png'.format(len(files))))
    os.remove(os.path.join(file_path, r'D_real', '{}.npy'.format(len(files))))
    os.remove(os.path.join(file_path, r'P_real', '{}.npy'.format(len(files))))
    os.remove(os.path.join(file_path, r'Force', '{}.npy'.format(len(files))))

data_path = r"processed_data_future"
train_path = r"train_future"
val_path = r"val_future"

for aim_path in [train_path, val_path]:
    if not os.path.exists(os.path.join(aim_path, r'label_P')):
        os.makedirs(os.path.join(aim_path, r'label_P'))
    if not os.path.exists(os.path.join(aim_path, r'label_F')):
        os.makedirs(os.path.join(aim_path, r'label_F'))
    if not os.path.exists(os.path.join(aim_path, r'label_D')):
        os.makedirs(os.path.join(aim_path, r'label_D'))

    if not os.path.exists(os.path.join(aim_path, r'label_P_20')):
        os.makedirs(os.path.join(aim_path, r'label_P_20'))
    if not os.path.exists(os.path.join(aim_path, r'label_F_20')):
        os.makedirs(os.path.join(aim_path, r'label_F_20'))
    if not os.path.exists(os.path.join(aim_path, r'label_D_20')):
        os.makedirs(os.path.join(aim_path, r'label_D_20'))

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

data_len = len(os.listdir(data_path))

count = 0
aim_path = train_path
for i in range(1, data_len + 1):
    if i == data_len:
        count = 0
        aim_path = val_path

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

        shutil.copyfile(os.path.join(file_path, r'label_D_20', '{}.npy'.format(j + 1)),
                        os.path.join(aim_path, r'label_D_20', '{}.npy'.format(count)))
        shutil.copyfile(os.path.join(file_path, r'label_P_20', '{}.npy'.format(j + 1)),
                        os.path.join(aim_path, r'label_P_20', '{}.npy'.format(count)))
        shutil.copyfile(os.path.join(file_path, r'label_F_20', '{}.npy'.format(j + 1)),
                        os.path.join(aim_path, r'label_F_20', '{}.npy'.format(count)))

        shutil.copyfile(os.path.join(file_path, r'cropped_ut_img', '{}.png'.format(j + 1)),
                        os.path.join(aim_path, r'cropped_ut_img', '{}.png'.format(count)))
        shutil.copyfile(os.path.join(file_path, r'{}'.format(i), '{}.png'.format(j + 1)),
                        os.path.join(aim_path, r'RGB_image', '{}.png'.format(count)))
        count = count + 1
