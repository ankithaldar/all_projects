#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''All models for classification'''


# imports
import timm
import torch.nn as nn

#    script imports
# imports


# constants
# constants


# classes
class EfficientNetB1Classifier(nn.Module):
  '''EfficientNet B1 Classifier'''

  def __init__(self, pretrained=True, drop_rate=0.3):
    super().__init__()
    self.model = timm.create_model(
      'efficientnet_b1',
      pretrained=pretrained,
      drop_rate=drop_rate,
      num_classes=0  # remove head
    )
    self.out = nn.Linear(self.model.num_features, 1)

  def forward(self, x):
    feats = self.model(x)
    return self.out(feats).squeeze(dim=1)  # (B,)



class ResNet34Classifier(nn.Module):
  '''ResNet 34 Classifier'''

  def __init__(self, pretrained=True):
    super().__init__()
    self.model = timm.create_model(
      'resnet34',
      pretrained=pretrained,
      num_classes=0
    )
    self.out = nn.Linear(self.model.num_features, 1)

  def forward(self, x):
    feats = self.model(x)
    return self.out(feats).squeeze(dim=1)

# classes
