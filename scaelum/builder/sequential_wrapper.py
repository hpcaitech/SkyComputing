#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch.nn as nn


class SequentialWrapper(nn.Sequential):
    """
    A wrapper class for nn.Sequential so that the nn.Sequential
    can handle multiple inputs
    """

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
