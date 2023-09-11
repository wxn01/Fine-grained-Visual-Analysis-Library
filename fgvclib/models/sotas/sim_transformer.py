# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import os.path as op
from scipy import ndimage
from torch.nn import Linear
from yacs.config import CfgNode

from fgvclib.criterions import LossItem
from fgvclib.models.sotas.sota import FGVCSOTA
from fgvclib.models.sotas import fgvcmodel

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


@fgvcmodel("SIM_Trans")
class SIM_Transformer(FGVCSOTA):
    def __init__(self, cfg: CfgNode, backbone: nn.Module, encoder: nn.Module, necks: nn.Module, heads: nn.Module,
                 criterions: nn.Module):
        super().__init__(cfg, backbone, encoder, necks, heads, criterions)
        self.num_classes = cfg.CLASS_NUM
        self.smoothing_value = self.args["smoothing_value"]
        self.zero_head = self.args["zero_head"]
        self.classifier = self.args["classifier"]

        self.part_head = nn.Sequential(
            nn.BatchNorm1d(self.args['hidden_size'] * 3),
            Linear(self.args['hidden_size'] * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            Linear(1024, cfg.CLASS_NUM),
        )
        self.pretrained_weight = self.args["pretrained_weight"]
        if op.exists(self.pretrained_weight):
            print(f"Loading pretraining weight in {self.pretrained_weight}.")
            if self.pretrained_weight.endswith('.npz'):
                self.load_from(np.load(self.pretrained_weight))
            else:
                self.load_state_dict(torch.load(self.pretrained_weight))

    def forward(self, x, labels=None, step=0, global_step=10000):
        part_tokens, hid = self.encoder(x)

        final_hid = torch.cat((hid[-2], hid[-1], part_tokens[:, 0]), dim=-1)
        part_logits = self.part_head(final_hid)

        if labels is not None:
            losses = list()
            if self.smoothing_value == 0:
                part_loss = self.criterions['cross_entropy_loss']['fn'](part_logits.view(-1, self.num_classes),
                                                                       labels.view(-1))
            else:
                part_loss = self.criterions['nll_loss_labelsmoothing']['fn'](part_logits.view(-1, self.num_classes),
                                                                            labels.view(-1))

            contrast_loss = self.criterions['con_lr_loss']['fn'](part_tokens[:, 0], labels.view(-1), step, global_step)
            losses.append(LossItem(name='contrast_loss', value=contrast_loss, weight=1.0))
            losses.append(LossItem(name='part_loss', value=part_loss, weight=1.0))

            return losses, part_logits
        else:
            return part_logits

    def load_from(self, weights):
        with torch.no_grad():
            self.encoder.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.encoder.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.encoder.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.encoder.encoder.part_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder.encoder.part_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.encoder.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.encoder.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.encoder.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.encoder.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.encoder.embeddings.hybrid:
                self.encoder.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.encoder.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.encoder.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.encoder.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
