import os
import shutil
import numpy

path = r"C:\Users\Paradox\Desktop\git\multimodel\multi_model_10_data"
for i in range(1, 11):
    file_path = os.path.join(path, "test{}".format(i))
    if not os.path.exists(os.path.join(file_path, r'label_D')):
        os.makedirs(os.path.join(file_path, r'label_D'))
    if not os.path.exists(os.path.join(file_path, r'label_P')):
        os.makedirs(os.path.join(file_path, r'label_P'))
    if not os.path.exists(os.path.join(file_path, r'label_F')):
        os.makedirs(os.path.join(file_path, r'label_F'))

    files = os.listdir(os.path.join(file_path, r'D_real'))

    # 删除predict
    shutil.rmtree(os.path.join(file_path, "D_expected"))
    shutil.rmtree(os.path.join(file_path, "P_expected"))
    if os.path.exists(os.path.join(file_path, r'{}'.format(i), '.DS_Store')):
        os.remove(os.path.join(file_path, r'{}'.format(i), '.DS_Store'))
    if os.path.exists(os.path.join(file_path, r'cropped_ut_img', '.DS_Store')):
        os.remove(os.path.join(file_path, r'cropped_ut_img', '.DS_Store'))

    for j in range(len(files)):
        shutil.copyfile(os.path.join(file_path, r'D_real', '{}.npy'.format(j + 1)),
                        os.path.join(file_path, r'label_D', '{}.npy'.format(j)))
        shutil.copyfile(os.path.join(file_path, r'P_real', '{}.npy'.format(j + 1)),
                        os.path.join(file_path, r'label_P', '{}.npy'.format(j)))
        shutil.copyfile(os.path.join(file_path, r'Force', '{}.npy'.format(j + 1)),
                        os.path.join(file_path, r'label_F', '{}.npy'.format(j)))

    os.remove(os.path.join(file_path, r'label_D', '0.npy'))
    os.remove(os.path.join(file_path, r'label_P', '0.npy'))
    os.remove(os.path.join(file_path, r'label_F', '0.npy'))
    os.remove(os.path.join(file_path, r'{}'.format(i), '{}.png'.format(len(files))))
    os.remove(os.path.join(file_path, r'cropped_ut_img', '{}.png'.format(len(files))))
    os.remove(os.path.join(file_path, r'D_real', '{}.npy'.format(len(files))))
    os.remove(os.path.join(file_path, r'P_real', '{}.npy'.format(len(files))))
    os.remove(os.path.join(file_path, r'Force', '{}.npy'.format(len(files))))
