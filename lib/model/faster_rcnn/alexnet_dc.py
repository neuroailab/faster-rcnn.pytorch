from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
from model.faster_rcnn.vgg16_dc_orig import load_model


def build_normalize():
	normalize = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
	normalize.weight.data.zero_()
	normalize.weight.data[0, 0].copy_(
			torch.FloatTensor([[1.0 / 0.225 / 255]]))
	normalize.weight.data[1, 1].copy_(
			torch.FloatTensor([[1.0 / 0.224 / 255]]))
	normalize.weight.data[2, 2].copy_(
			torch.FloatTensor([[1.0 / 0.229 / 255]]))
	normalize.bias.data.zero_()
	for p in normalize.parameters():
		p.requires_grad = False
	return normalize


class alexnet_dc(_fasterRCNN):
  def __init__(
      self, classes, pretrained=False, class_agnostic=False,
      fix_layers=15):
    self.model_path = 'data/pretrained_model/alexnet_dc.pth'
    self.dout_base_model = 256
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.fix_layers = fix_layers

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    alexnet = load_model(self.model_path)

    alexnet.classifier = nn.Sequential(*list(alexnet.classifier._modules.values()))

    # not using the last maxpool layer
    RCNN_base_layers = [build_normalize()]
    RCNN_base_layers.extend(list(alexnet.sobel._modules.values()))
    RCNN_base_main = list(alexnet.features._modules.values())[:-1]
    RCNN_base_layers.extend(RCNN_base_main)
    self.RCNN_base = nn.Sequential(*RCNN_base_layers)

    self.RCNN_top = alexnet.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    #self.RCNN_base.apply(set_bn_fix)
    #self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

