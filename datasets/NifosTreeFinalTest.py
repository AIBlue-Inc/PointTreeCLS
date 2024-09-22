'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import glob
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def load_data(data_path, partition, class_names):
    DATA_DIR = os.path.join(data_path, '*', partition)
    all_data = []
    all_label = []
    file_list = glob.glob(os.path.join(DATA_DIR, '*.xyz'))
    for file_path in file_list:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            data = []
            for line in lines:
                values = line.strip().split()
                # Assuming each line contains X, Y, Z coordinates
                if len(values) >= 3:
                    data.append([float(values[0]), float(values[1]), float(values[2])])
            if len(data) > 0:
                data = np.array(data)
                all_data.append(data)
                # You can extract the label from the file name or any other source
                # For example, assuming the file name is in the format 'Densi_X_2048.xyz'
                label = class_names.index(file_path.split('/')[-3])
                all_label.append(label)

    # Convert data and label lists to numpy arrays
    return all_data, all_label, file_list


@DATASETS.register_module()
class NifosTree4Test(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset

        self.cat = ['Densi', 'gal', 'gul', 'Koraiensis', 'Larix', 'obtusa', 'sang', 'sin']
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.list_of_points, self.list_of_labels, self.list_of_files = load_data(self.root, split, self.cat)

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, index):
        points, label, file = self.list_of_points[index], self.list_of_labels[index], self.list_of_files[index]
        pt_idxs = np.arange(0, points.shape[0])   # 1024
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'NifosTree4Test', 'sample', (current_points, label, file)