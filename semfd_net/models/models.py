import torch
import torch.nn as nn
import torchvision.models as models
from resnest.torch import resnest101,resnest50
import timm

class Classifier(nn.Module):
    def __init__(self, backbone):
        super(Classifier, self).__init__()
        
        if backbone=='vgg16':
            self.backbone = models.vgg16(pretrained=True)
        elif backbone =='densenet121':
            self.backbone = models.densenet121(pretrained=True)
        elif backbone=='resnet50':
            self.backbone = models.resnet50(pretrained=True)
        elif backbone=='resnet101':
            self.backbone = models.resnet101(pretrained=True)
        elif backbone=='resnest50':
            self.backbone = resnest50(pretrained=True)        
        elif backbone=='resnest101':
            self.backbone = resnest101(pretrained=True)
        elif backbone=='vit':
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        else:
            raise Exception("Invalid Model!")
    
        self.convnet = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False), 
                   nn.Linear(1000, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False),
                   nn.Linear(512,27))
   
    def forward(self, x):
        x = self.backbone(x)
        x = self.convnet(x)
        return x
    
class MetaLearner(nn.Module):
    def __init__(self):
      super(MetaLearner, self).__init__()

      self.meta_learner = nn.Sequential(nn.BatchNorm1d(162),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.3),
                                nn.Linear(162, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.3),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.3),
                                nn.Linear(256,128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.3),
                                nn.Linear(128, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.3),
                                nn.Linear(64, 27))

    def forward(self, x):
        x.squeeze_(1)
        x = self.meta_learner(x)
        return x