import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image

class BenchmarkDataset(torch.utils.data.Dataset):

    def __init__(self, CFG, mode):
        super(BenchmarkDataset, self).__init__()
        self.CFG = CFG
        self.mode = mode
        self.dataset_path = os.path.join(CFG.dataset_path, mode)
        self.resize = 224 if CFG.name=="benchmark" and CFG.backbone=="vit" else 256

        self.transforms = transforms.Compose([
            transforms.Resize((self.resize,self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.469, 0.536, 0.369],
                            std=[0.260, 0.244, 0.282])
        ])

        self.datas, self.num_classes = self.get_image_data(self.dataset_path)

    def get_image_data(self, dataset_path):
        print(f"Scanning {self.mode} data...")
        datas = []
        idx = 0
        for dis_cat in sorted(os.listdir(dataset_path)):
            for imgPath in os.listdir(os.path.join(dataset_path, dis_cat)):
                filePath = os.path.join(dataset_path, dis_cat, imgPath)
                datas.append([filePath,idx])
            idx += 1
        print(f"Meta-data for {self.mode} split collected.")
        return datas, idx

    def __len__(self): 
        return len(self.datas)

    def __getitem__(self, index):
        
        label = self.datas[index][1]
        image = Image.open(self.datas[index][0]).convert('RGB')
            
        if self.transforms:
            image = self.transforms(image)

        return image, label