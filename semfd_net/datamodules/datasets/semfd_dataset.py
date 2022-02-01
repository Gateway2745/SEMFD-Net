import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image

class SEMFD_Dataset(torch.utils.data.Dataset):

    def __init__(self, datas, mode, num_classes):
        super(SEMFD_Dataset, self).__init__()
        self.datas = datas
        self.mode = mode
        self.num_classes = num_classes

    def __len__(self): 
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index][0], self.datas[index][1]