#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import time
from collections import OrderedDict
from typing import List, Dict

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn

# global params
GPU = torch.cuda.is_available()


def synchronize():
    if GPU:
        torch.cuda.synchronize()


def get_time():
    synchronize()
    return time.time()


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


def load_weights(model: nn.Module, state_dict: List[Dict]):
    # model.modules() gives a generator
    # the first element in the list is the whole module
    # thus exclude it when loading weights
    modules = list(model.modules())[1]

    error_msg = "Weights do not match the model, model has {} modules while state dict has {}".format(
        len(modules), len(state_dict)
    )
    assert len(modules) == len(state_dict), error_msg

    # load weights
    for idx, mod in enumerate(modules):
        mod.load_state_dict(state_dict[idx])


def get_state_dict(model: nn.Module) -> List[Dict]:
    modules = list(model.modules())[1]
    module_weights = [mod.state_dict() for mod in modules]
    module_weights = [weights_to_cpu(weights) for weights in module_weights]
    return module_weights


def weights_to_cpu(state_dict):
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    return param_rrefs


def count_params(model, to_console=False):
    num_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    num_grad_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    )

    if to_console:
        print("Number of parameters: {:.5g} M".format(num_params))
        print("Number of parameters requiring grad: {:.5g} M".format(num_grad_params))

    return num_params, num_grad_params


def generate_worker_name(rank):
    return "worker{}".format(rank)
