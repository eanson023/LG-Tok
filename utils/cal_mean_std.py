import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm

#################################################################################
#                                Calculate Mean Std                             #
#################################################################################
def mean_variance(data_dir, save_dir, joints_num):
    data_list = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            data = np.load(pjoin(root, file))
            if np.isnan(data).any():
                print(file)
                continue
            data_list.append(data[:, :4+(joints_num-1)*3])

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
    # data_dir1 = 'datasets/HumanML3D/new_joint_vecs/'
    # save_dir1 = 'datasets/HumanML3D/'
    # mean, std = mean_variance(data_dir1, save_dir1, 22)

    # data_dir2 = 'datasets/KIT-ML/new_joint_vecs/'
    # save_dir2 = 'datasets/KIT-ML/'
    # mean2, std2 = mean_variance(data_dir2, save_dir2, 21)

    # data_dir3 = 'dataset/CMP/new_joint_vecs/'
    # save_dir3 = 'dataset/CMP/'
    # mean2, std2 = mean_variance(data_dir3, save_dir3, 22)
    data_dir3 = 'dataset/Motion-X/vector_263/'
    save_dir3 = 'dataset/Motion-X/'
    mean2, std2 = mean_variance(data_dir3, save_dir3, 22)
    # res = np.load('datasets/HumanML3D/Mean.npy')
    # print(res.shape)