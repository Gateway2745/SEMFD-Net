import torch
import torch.nn.functional as F
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from semfd_net.datamodules.benchmark_datamodule import BenchmarkDataModule
from semfd_net.datamodules.semfd_datamodule import SEMFD_DataModule

from semfd_net.models.models import Classifier, MetaLearner
from semfd_net.losses import FocalLoss

class FoliarDiseaseClassification(pl.LightningModule):
    def __init__(self, CFG):
        super(FoliarDiseaseClassification, self).__init__()
        self.CFG=CFG
        if CFG.name == "benchmark":
            self.model = Classifier(CFG["backbone"])
            self.loss_fn = F.cross_entropy
        elif CFG.name == "semfd_net":
            self.model = MetaLearner()
            self.loss_fn = FocalLoss()
        else:
            raise ValueError(f'model "{CFG.training.model}" does not exist! Must be one of "benchmark" / "semfd_net"')
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.best_val_acc = -1

    def forward(self, x):
        out = self.model(x)
        return out
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.CFG)

    def training_step(self, batch, batch_idx):
        *inps, y = batch
        y = y.view(-1,)
        out = self.model(*inps)
        loss = self.loss_fn(out, y)
        self.log("train_loss",loss,sync_dist=True,rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        *inps, y = batch
        y = y.view(-1,)
        out = self.model(*inps)
        loss = self.loss_fn(out, y).item()
        y_pred = torch.argmax(out, axis=1)
        
        self.log("val_loss_epoch", loss, logger=True, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log('val_acc_step', self.val_accuracy(y_pred, y), logger=False)

    def validation_epoch_end(self, val_step_outputs):
        self.log('val_acc_epoch', self.val_accuracy.compute(), logger=False, prog_bar=True)
        self.logger.log_metrics({"val_acc_epoch": self.val_accuracy.compute().item()}, step = self.trainer.current_epoch)
        self.best_val_acc = max(self.best_val_acc, self.val_accuracy.compute().item())
        self.val_accuracy.reset()

    def test_step(self, batch, batch_idx):
        *inps, y = batch
        y = y.view(-1,)
        out = self.model(*inps)
        loss = self.loss_fn(out, y).item()
        y_pred = torch.argmax(out, axis=1)

    def test_epoch_end(self, test_step_outputs):
        self.log('test_acc', self.test_accuracy.compute(), logger=False, prog_bar=True)
        self.logger.log_metrics({"test_acc": self.test_accuracy.compute().item()}, step = self.trainer.current_epoch)
        self.test_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.CFG.training.learning_rate)
        return optimizer
        

def train(CFG):
  if CFG.name == "benchmark":
    dm = BenchmarkDataModule(CFG)
  elif CFG.name == "semfd_net":
    dm = SEMFD_DataModule(CFG)
    
  dm.prepare_data()

  for run_no in range(1,CFG.training.num_runs+1):
      if CFG.deterministic.set:
          seed_everything(CFG.deterministic.seed, workers=True)

      model = FoliarDiseaseClassification(CFG)
      dm.setup()
      mlf_logger = pl_loggers.mlflow.MLFlowLogger(experiment_name=CFG.name, run_name = f'run={run_no}', save_dir=CFG.training.save_dir)
      checkpoint_callback = ModelCheckpoint(dirpath=None,
                                          monitor='val_acc_epoch',
                                          save_top_k=1 if CFG.training.save_dir else 0,
                                          save_last=True if CFG.training.save_dir else False,
                                          save_weights_only=True,
                                          filename='{epoch:02d}-{val_acc_epoch:.4f}',
                                          verbose=False,
                                          mode='max')

      trainer = Trainer(
          max_epochs=CFG.training.num_epochs,
          num_nodes=CFG.num_nodes,
          gpus=CFG.gpus,
          precision=32,
          callbacks=[checkpoint_callback],
          logger = mlf_logger,
          weights_summary='top',
          log_every_n_steps=4,
          deterministic=CFG.deterministic.set,
          accelerator = "ddp" if CFG.gpus and len(CFG.gpus)>1 else None,
      )

      trainer.fit(model,dm)

      trainer.test(ckpt_path='best')
