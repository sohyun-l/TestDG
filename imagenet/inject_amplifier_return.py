import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F

import torch.nn as nn

class AMPLIFIERInjectedLinearReturn(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, r2 = 64):
        super().__init__()

        self.linear_amplifier = nn.Linear(in_features, out_features, bias)
        self.amplifier_down2 = nn.Linear(in_features, r2, bias=False)
        self.amplifier_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale1 = 1.0
        self.scale2 = 1.0


        nn.init.normal_(self.amplifier_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.amplifier_up2.weight)

    def forward(self, input):
        return self.linear_amplifier(input) + self.amplifier_up2(self.amplifier_down2(input)) * self.scale2



def inject_trainable_amplifier_return(
    model: nn.Module,
    target_replace_module: List[str] = ["CrossAttention", "Attention"],
    r: int = 4,
    r2: int = 16,
):

    require_grad_params = []
    names = []

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = AMPLIFIERInjectedLinearReturn(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        r2,
                    )
                    _tmp.linear_amplifier.weight = weight
                    if bias is not None:
                        _tmp.linear_amplifier.bias = bias

                    # switch the module
                    _module._modules[name] = _tmp


                    require_grad_params.extend(
                        list(_module._modules[name].amplifier_up2.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].amplifier_down2.parameters())
                    )
                    _module._modules[name].amplifier_up2.weight.requires_grad = True
                    _module._modules[name].amplifier_down2.weight.requires_grad = True                    
                    names.append(name)

    return require_grad_params, names