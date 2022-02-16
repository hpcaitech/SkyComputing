#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from dllb.builder import build_layer
from torch import Tensor


class ParameterServer(nn.Module):
    def __init__(self, model_config: list) -> None:
        super(ParameterServer, self).__init__()
        self._model_config = model_config
        self.module_list = nn.ModuleList()

        self._build_model()

    def _build_model(self) -> None:
        for cfg in self._model_config:
            cfg_copy = cfg.copy()
            layer_type = cfg_copy.pop("layer_type")
            layer = build_layer(layer_type, **cfg_copy)
            self.module_list.append(layer)

    def load_weights_from_file(self, checkpoint: str) -> None:
        self.module_list.load_state_dict(torch.load(checkpoint))

    def save_weights_to_file(self, checkpoint: str) -> None:
        torch.save(self.module_list.state_dict(), checkpoint)

    def update_weights(self, state_dict: OrderedDict, idx: int) -> None:
        self.module_list[idx].load_state_dict(state_dict)

    def get_state_dict(self, idx: int) -> Dict[str, Tensor]:
        return self.module_list[idx].state_dict()
