#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import abc
import os
import dllb.utils as dutils
import torch
import torch.distributed.rpc as rpc
from dllb.builder import build_module_from_cfg, build_layer
from dllb.dataset import BaseGenerator
from .estimator import Estimator
from .worker_manager import WorkerManager

if os.getenv("STIMULATE") is not None:
    from dllb.stimulator import Stimulator


class BaseBenchmarker(object):
    """
    A base class for benchmarking objects
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def benchmark(self):
        raise NotImplementedError("not implemented yet")


class DeviceBenchmarker(BaseBenchmarker):
    def __init__(
        self,
        worker_manager: WorkerManager,
        data_generator: BaseGenerator,
        model_config: dict,
        iterations: int,
        dtype: str = None,
    ):
        super(DeviceBenchmarker, self).__init__()
        self._worker_manager = worker_manager
        self._model_config = model_config
        self._data_generator = data_generator
        self._iterations = iterations
        self._dtype = dtype
        # If stimulate on HPC, then use stimulator to slowdown
        if os.getenv("STIMULATE") is not None:  # TODO: Is the usage correct?
            self._stimulator = Stimulator(self._worker_manager.size)

    @staticmethod
    def local_benchmark(rank, data, model_cfg, module_wrapper_cfg, iterations, dtype):
        """
        A method to wrap the benchmarking code
        """
        model = build_module_from_cfg(
            rank=rank, model_cfg=model_cfg, module_wrapper_cfg=module_wrapper_cfg
        )

        device = next(model.parameters()).device
        # print(
        #     'rank : {}: device: {}, model gpu index: {}'.format(rank, device, model.gpu_index))

        time = Estimator.benchmark_speed(
            model=model, data=data, device=device, iterations=iterations, dtype=dtype
        )
        avai_mem = model.detect_mem(destroy_module=True)
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return time, avai_mem

    def benchmark(self):
        """
        Run benchmarking to test the computational speed and available memory
        of the local or remote
        """

        # init results
        results = dict()

        # get data
        data = self._data_generator.generate()

        # benchmarking on different devices
        result_queue = []
        for worker in self._worker_manager.worker_pool:
            rank = worker.rank
            worker_name = dutils.generate_worker_name(rank)
            module_wrapper_cfg = worker.extra_config.copy()

            if rank == 0:
                # run locally
                time, avai_mem = self.local_benchmark(
                    rank,
                    data,
                    self._model_config,
                    module_wrapper_cfg,
                    self._iterations,
                    self._dtype,
                )
                result_queue.append((worker_name, time, avai_mem))
            else:
                res = rpc.rpc_async(
                    to=worker_name,
                    func=DeviceBenchmarker.local_benchmark,
                    args=(
                        rank,
                        data,
                        self._model_config,
                        module_wrapper_cfg,
                        self._iterations,
                        self._dtype,
                    ),
                )
                result_queue.append((worker_name, res))

        for res in result_queue:
            worker_name = res[0]

            if len(res) == 2:
                time, avai_mem = res[1].wait()
            else:
                time, avai_mem = res[1], res[2]

            if os.getenv("STIMULATE") is not None:
                # int(res[0].lstrip("worker")) get the original rank
                time *= self._stimulator.c_slowdown[int(res[0].lstrip("worker"))]
                avai_mem /= self._stimulator.m_slowdown[int(res[0].lstrip("worker"))]

            results[worker_name] = dict(time=time, avai_mem=avai_mem)

        return results


class ModelBenchmarker(BaseBenchmarker):
    def __init__(
        self,
        model_config: dict,
        data_generator: BaseGenerator,
        device: str,
        dtype: str = None,
        param_scale: int = 2,
    ):
        super(ModelBenchmarker, self).__init__()
        self._model_config = model_config
        self._data_generator = data_generator
        self._device = device
        self._dtype = dtype
        self._param_scale = param_scale

    @property
    def model_config(self):
        return self._model_config

    def benchmark(self):
        flops_list = []
        mem_list = []

        # measure flops of each layer
        data = self._data_generator.generate()

        # NOTE: this only applies to BERT since the single machine cannot host such a large model and will cause OOM
        # TODO: Remove this is you wish to use this framework for other models
        num_encoder_layer = int((len(self._model_config) - 3) / 3)
        model_cfg = self._model_config[:4] + self._model_config[-2:]

        for idx, layer_cfg in enumerate(model_cfg):
            # build layer
            layer_cfg_copy = layer_cfg.copy()
            layer_type = layer_cfg_copy.pop("layer_type")
            layer = build_layer(layer_type, **layer_cfg_copy)

            # get flops and mem usage
            output, flops, mem_usage = Estimator.benchmark_model(
                model=layer,
                data=data,
                device=self._device,
                dtype=self._dtype,
                param_scale=self._param_scale,
            )

            # remove layer to save RAM
            del layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # override input
            data = output

            # log
            flops_list.append(flops)
            mem_list.append(mem_usage)

        # NOTE: this only applies to BERT since the single machine cannot host such a large model and will cause OOM
        # TODO: Remove this is you wish to use this framework for other models
        flops_list = (
            [flops_list[0]] + flops_list[1:4] * num_encoder_layer + flops_list[-2:]
        )
        mem_list = [mem_list[0]] + mem_list[1:4] * num_encoder_layer + mem_list[-2:]
        return flops_list, mem_list
