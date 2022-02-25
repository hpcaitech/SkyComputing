#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch.nn as nn


class Registry(object):
    def __init__(self, name: str):
        self.name = name
        self._registry = dict()

    def register_module(self, module_class):
        module_name = module_class.__name__
        assert module_name not in self._registry
        self._registry[module_name] = module_class

    def get_module(self, module_name: str, include_torch=True):
        if module_name in self._registry:
            return self._registry[module_name]
        elif include_torch and hasattr(nn, module_name):
            return getattr(nn, module_name)
        else:
            raise NameError("Module {} not found".format(module_name))


LAYER = Registry("layer")
DATASET = Registry("dataset")
HOOKS = Registry("hook")
DATA_GENERATOR = Registry("data_generator")
