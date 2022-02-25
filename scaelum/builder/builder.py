#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from scaelum.registry import DATA_GENERATOR, DATASET, HOOKS, LAYER, Registry
from torch.utils.data import DataLoader

from .module_wrapper import ModuleWrapper
from .sequential_wrapper import SequentialWrapper


def build_from_registry(module_name: str, registry: Registry, *args, **kwargs):
    mod = registry.get_module(module_name)
    return mod(*args, **kwargs)


def build_layer(module_name: str, *args, **kwargs):
    return build_from_registry(module_name, LAYER, *args, **kwargs)


def build_hook(module_name: str, *args, **kwargs):
    return build_from_registry(module_name, HOOKS, *args, **kwargs)


def build_data_generator(module_name: str, *args, **kwargs):
    return build_from_registry(module_name, DATA_GENERATOR, *args, **kwargs)


def build_module_from_cfg(rank, model_cfg: list, module_wrapper_cfg: dict):
    layers = []

    for layer_cfg in model_cfg:
        layer_cfg_copy = layer_cfg.copy()
        layer_type = layer_cfg_copy.pop("layer_type")
        layer = build_layer(layer_type, **layer_cfg_copy)
        layers.append(layer)
    module = SequentialWrapper(*layers)
    module_wrapper_cfg["record_forward_time"] = True
    module = ModuleWrapper(rank=rank, module=module, **module_wrapper_cfg)

    return module


def build_dataloader_from_cfg(dataset_cfg, dataloader_cfg):
    dataset_type = dataset_cfg.pop("type")
    dataset = build_from_registry(dataset_type, DATASET, **dataset_cfg)
    dataloader = DataLoader(dataset, **dataloader_cfg)

    return dataloader
