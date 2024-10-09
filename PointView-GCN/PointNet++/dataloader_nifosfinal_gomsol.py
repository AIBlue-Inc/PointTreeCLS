import numpy as np
import glob
import torch.utils.data
import torch

class SinglePointFinalGomsol(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, npoint=2048, test_mode=False, num_models=0, num_views=20):
        self.classnames = ['Densi', 'gul', 'Koraiensis', 'Larix', 'obtusa', 'sang', 'sin', 'thunbergii']

        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.npoints = npoint
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []

        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.xyz'))
            self.filepaths.extend(all_files)

    def __len__(self):
        # print(len(self.filepaths))
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        point_set = np.loadtxt(self.filepaths[idx])
        return (point_set ,class_id)
