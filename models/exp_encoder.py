from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from dataclasses import dataclass




class ExpEncoder(ModelMixin):
    def __init__(
        self,
        input_size: int = 64,
        hidden_sizes=None,
        output_size: int = 768,

    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 256, 512]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            # 输出层不加激活函数
        x = self.layers[-1](x)

        return x
