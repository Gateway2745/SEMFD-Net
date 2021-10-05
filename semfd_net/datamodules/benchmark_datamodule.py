import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import glob

from .datasets.benchmark_dataset import BenchmarkDataset

class BenchmarkDataModule(pl.LightningDataModule):
    def __init__(self, CFG):
        super(BenchmarkDataModule, self).__init__()
        self.CFG=CFG

    def prepare_data(self):
        pass

    def setup(self):
        self.train_dataset = BenchmarkDataset(self.CFG, mode="train")
        self.val_dataset = BenchmarkDataset(self.CFG, mode="val")
        self.test_dataset = BenchmarkDataset(self.CFG, mode="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.CFG.training.train_dataloader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.CFG.training.val_dataloader)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.CFG.training.val_dataloader)

    def teardown(self,stage):
        import gc
        del self.train_dataset
        del self.val_dataset
        del self.train_dataset
        del self.val_dataset
        gc.collect()