import os
import glob
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class_names = ['Densi', 'gal', 'gul', 'Koraiensis', 'Larix', 'obtusa', 'sang', 'sin']


def load_data(partition, num_points):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../data', f'240916_total_{num_points}_xyz', '*', partition)
    all_data = []
    all_label = []
    file_list = glob.glob(os.path.join(DATA_DIR, '*.xyz'))
    assert len(file_list) > 0, f"No data found in {DATA_DIR}"
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
                all_data.append(data)
                # You can extract the label from the file name or any other source
                # For example, assuming the file name is in the format 'Densi_X_2048.xyz'
                label = class_names.index(file_path.split(os.sep)[-3])
                all_label.append(label)

    # Convert data and label lists to numpy arrays
    all_data = np.array(all_data, dtype=np.float32)
    all_label = np.array(all_label, dtype=np.int64)
    return all_data, all_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class NifosTreeFinal(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition, num_points)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = NifosTreeFinal(1024)
    test = NifosTreeFinal(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
    from torch.utils.data import DataLoader

    train_loader = DataLoader(NifosTreeFinal(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;label shape: {label.shape}")

    train_set = NifosTreeFinal(partition='train', num_points=1024)
    test_set = NifosTreeFinal(partition='valid', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
