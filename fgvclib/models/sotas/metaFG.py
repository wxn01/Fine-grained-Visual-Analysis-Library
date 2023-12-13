
import torch
from torch import nn
import numpy as np
from yacs.config import CfgNode
import torch.nn.functional as F

from fgvclib.configs.utils import turn_list_to_dict as tltd
from fgvclib.models.sotas.sota import FGVCSOTA
from fgvclib.models.sotas import fgvcmodel
from fgvclib.criterions import LossItem
from fgvclib.transforms import MixUpCutMix

@fgvcmodel("MetaFG")
class MetaFG(FGVCSOTA):
    r"""
        Code of "MetaFormer: A Unified Meta Framework for Fine-Grained Recognition".
        Link: https://github.com/dqshuai/MetaFormer
    """

    def __init__(self, cfg: CfgNode, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module, criterions: nn.Module):
        super().__init__(cfg, backbone, encoder, necks, heads, criterions)
        num_classes = cfg.CLASS_NUM
        self.add_meta = cfg.ADD_META
        self.mixup_fn = MixUpCutMix(cfg=tltd(cfg.ARGS), num_classes= num_classes)

    def forward(self, x, targets=None, meta=None):
        if self.training:
            x, targets = self.mixup_fn(x,targets)

        losses = list()
        if self.add_meta:
            out = self.backbone(x,meta)
        else:
            out = self.backbone(x)
        out = self.heads(tuple([out]))

        if self.training:
            losses.append(LossItem(name="soft_target_cross_entropy_loss", 
                                   value=self.criterions['soft_target_cross_entropy_loss']['fn'](F.log_softmax(x, dim=-1), targets), 
                                   weight=self.criterions['soft_target_cross_entropy_loss']['w'])) 

            return out, losses

        return out
    