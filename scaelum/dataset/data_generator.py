#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import abc

import torch
from scaelum.builder import build_dataloader_from_cfg
from scaelum.registry import DATA_GENERATOR


class BaseGenerator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate(self):
        pass


@DATA_GENERATOR.register_module
class RandomTensorGenerator(BaseGenerator):
    def __init__(self, generator_cfg):
        self.generator_cfg = generator_cfg

    def generate(self):
        return torch.rand(**self.generator_cfg)


@DATA_GENERATOR.register_module
class DataloaderGenerator(BaseGenerator):
    def __init__(self, generator_cfg):
        self.dataloader = build_dataloader_from_cfg(**generator_cfg)
        self.generator = iter(self.dataloader)

    def generate(self):
        return next(iter(self.dataloader))[0]
