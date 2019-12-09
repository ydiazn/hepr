'''
Custom datasets
'''

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from glob import glob


class HeadMRIDataset(Dataset):
    def __init__(self, root_dir, size=(256, 256)):
        self.files = glob(root_dir + '*.bmp')
        self.size = size
    def __len__(self,):
        return len(self.files)
    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        img = img.transpose(2, 0, 1)
        return img