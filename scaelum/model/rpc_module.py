#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import List, Dict

import torch.distributed.rpc as rpc
import torch.nn as nn
from scaelum import utils
from scaelum.builder import ModuleWrapper, build_module_from_cfg


class BaseModule(nn.Module):
    def __init__(
        self,
        rank,
        model_cfg,
        sequential_wrapper_cfg,
    ):
        super(BaseModule, self).__init__()
        self.rank = rank
        self.model_cfg = model_cfg
        self.sequential_wrapper_cfg = sequential_wrapper_cfg
        self.module = self._build_module()

    def _forward(self, *args):
        raise Exception("not implemented")

    def forward(self, *args, **kwargs):
        output = self._forward(*args, **kwargs)
        return output

    def _build_module(self):
        raise NotImplementedError("not implemented")

    def load_weights(self, state_dict: List[Dict]) -> None:
        raise NotImplementedError("not implemented")

    def get_state_dict(self) -> List[Dict]:
        raise NotImplementedError("not implemented")

    def parameter_rrefs(self) -> List:
        raise NotImplementedError("not implemented")


class LocalModule(BaseModule):
    def _forward(self, *args):
        res = self.module(*args)

        if isinstance(res, tuple) or isinstance(res, list):
            return res
        else:
            return (res,)

    def _build_module(self):
        module = build_module_from_cfg(
            rank=self.rank,
            model_cfg=self.model_cfg,
            module_wrapper_cfg=self.sequential_wrapper_cfg,
        )
        return module

    def load_weights(self, state_dict: List[Dict]) -> None:
        utils.load_weights(self.module, state_dict)
        self.module._move_module_to_cuda()

    def get_state_dict(self) -> List[Dict]:
        return utils.get_state_dict(self.module)

    def parameter_rrefs(self) -> List:
        return utils.parameter_rrefs(self.module)


class RemoteModule(BaseModule):
    def _forward(self, *args):
        # must return as a tuple for consistency
        res = rpc.remote(
            "worker{}".format(self.rank),
            ModuleWrapper.forward,
            args=[self.module] + list(args),
        )
        return (res,)

    def _build_module(self):
        module = rpc.remote(
            "worker{}".format(self.rank),
            build_module_from_cfg,
            args=(self.rank, self.model_cfg, self.sequential_wrapper_cfg),
        )
        return module

    def load_weights(self, state_dict: List[Dict]) -> None:
        utils.remote_method(utils.load_weights, self.module, state_dict)
        utils.remote_method(ModuleWrapper._move_module_to_cuda, self.module)

    def get_state_dict(self) -> List[Dict]:
        return utils.remote_method(utils.get_state_dict, self.module)

    def parameter_rrefs(self) -> List:
        return utils.remote_method(utils.parameter_rrefs, self.module)
