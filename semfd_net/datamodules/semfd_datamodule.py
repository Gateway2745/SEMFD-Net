import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from tqdm import tqdm
import glob
from PIL import Image
import copy
import pickle

from semfd_net.datamodules.datasets.semfd_dataset import SEMFD_Dataset

# paths to trained base models
MODEL_PATHS = ['densenet121.ckpt',
               'resnet50.ckpt',
               'resnet101.ckpt',
               'resnest50.ckpt',
               'resnest101.ckpt',
               'vit.ckpt'
              ]

class SEMFD_DataModule(pl.LightningDataModule):
    def __init__(self, CFG):
        super(SEMFD_DataModule, self).__init__()
        self.CFG=CFG

    def prepare_data(self):
        data_transforms = [transforms.Compose([transforms.Resize((256, 256)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.469, 0.536, 0.369],
                          std=[0.260, 0.244, 0.282])]),
                          
                          transforms.Compose([transforms.Resize((224, 224)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.469, 0.536, 0.369],
                          std=[0.260, 0.244, 0.282])])]

        with torch.no_grad():
            for mode in ["val", "test"]:
              if os.path.exists(f"./ensemble_data_{mode}.pkl"):
                continue

              self.CFG["name"] = "benchmark"
              self.dataset_path = self.CFG["dataset_path"]
              self.cfgs = [copy.deepcopy(CFG) for _ in range(6)]
              self.cfgs[0]["backbone"] = "densenet121"
              self.cfgs[1]["backbone"] = "resnet50"
              self.cfgs[2]["backbone"] = "resnet101"
              self.cfgs[3]["backbone"] = "resnest50"
              self.cfgs[4]["backbone"] = "resnest101"
              self.cfgs[5]["backbone"] = "vit"
              print("loading pretrained models...!")
              from semfd_net.train import FoliarDiseaseClassification
              self.models = [FoliarDiseaseClassification.load_from_checkpoint(MODEL_PATHS[i], CFG=self.cfgs[i]).eval().cuda() for i in range(6)]
              print("loaded pretrained models!")
              self.CFG["name"] = "semfd_net"

              print(f"Pre-computing {mode} features...")
              idx = 0
              datas = []
              for dis_cat in tqdm(sorted(os.listdir(os.path.join(self.dataset_path, mode)))):
                for imgPath in os.listdir(os.path.join(self.dataset_path, mode, dis_cat)):
                    filePath = os.path.join(self.dataset_path, mode, dis_cat, imgPath)

                    image = Image.open(filePath).convert('RGB')

                    image1 = data_transforms[0](image).unsqueeze(0).cuda()
                    image2 = data_transforms[1](image).unsqueeze(0).cuda()

                    features = [m(image1) for m in self.models[:-1]] + [self.models[-1](image2)]
                    features = torch.cat(features,dim=1)

                    datas.append([features.cpu().numpy(),idx])
                idx += 1
              self.num_classes = idx

              with open(f'ensemble_data_{mode}.pkl', 'wb') as f:
                pickle.dump(datas, f)

    def setup(self):
        with open('./ensemble_data_val.pkl','rb') as f:
          datas = pickle.load(f)
          self.val_dataset = SEMFD_Dataset(datas, "val", 27)

        with open('./ensemble_data_test.pkl','rb') as f:
          datas = pickle.load(f)
          self.test_dataset = SEMFD_Dataset(datas, "test", 27)

    def train_dataloader(self):
        return DataLoader(self.val_dataset, **self.CFG.training.train_dataloader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.CFG.training.val_dataloader)

    def teardown(self,stage):
        import gc
        del self.train_dataset
        del self.val_dataset
        del self.test_dataset
        gc.collect()

if __name__ == "__main__":
  from omegaconf import OmegaConf
  CFG = OmegaConf.load("../../configs/config.yaml")
  dm = SEMFD_DataModule(CFG)
  dm.prepare_data()