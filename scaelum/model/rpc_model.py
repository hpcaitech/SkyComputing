#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch.nn as nn
from scaelum.dynamics import WorkerManager

from .rpc_module import LocalModule, RemoteModule

try:
    from torch.distributed.rpc import PyRRef as RpcRef
except ImportError:
    from torch.distributed.rpc import RRef as RpcRef


class RpcModel(nn.Module):
    def __init__(self, worker_manager: WorkerManager):
        super(RpcModel, self).__init__()
        self.worker_manager = worker_manager
        self.model = self._build_model()
        assert isinstance(self.model, nn.ModuleList), "model must be iterable"

    def _build_model(self):
        # init model
        model = nn.ModuleList()

        for worker in self.worker_manager.worker_pool:
            if worker.rank == 0:
                module = LocalModule(
                    rank=worker.rank,
                    model_cfg=worker.model_config,
                    sequential_wrapper_cfg=worker.extra_config,
                )
            else:
                module = RemoteModule(
                    rank=worker.rank,
                    model_cfg=worker.model_config,
                    sequential_wrapper_cfg=worker.extra_config,
                )
            model.append(module)

        return model

    def forward(self, *args):
        # Handle input in the case of List[List[Tensor]]
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]

        for idx, layer in enumerate(self.model):
            args = layer(*args)

        result = args[0]
        if isinstance(result, RpcRef):
            result = result.to_here()[0]
        return result

    def parameter_rrefs(self):
        remote_params = []

        for layer in self.model:
            remote_params.extend(layer.parameter_rrefs())

        return remote_params
