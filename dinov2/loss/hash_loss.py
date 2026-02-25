import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import NoneType, nn
from functools import partial
from torch.nn.init import trunc_normal_

class HashLoss(nn.Module):
    def __init__(self, do_loss_hash_reg=False,):
        super().__init__()
        self.do_loss_hash_reg = do_loss_hash_reg

    def forward(self, u_hash):
        hash_loss_dict = {}
        if self.do_loss_hash_reg:
            loss_hash_reg = (u_hash.abs() - 1).abs().mean()
            hash_loss_dict['loss_hash_reg'] = loss_hash_reg

        return hash_loss_dict

