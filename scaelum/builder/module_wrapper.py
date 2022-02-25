#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import subprocess
import time

import psutil
import torch
import torch.nn as nn
from scaelum import utils as dutils
from scaelum.logger import Logger
from scaelum.timer import DistributedTimer

from .sequential_wrapper import SequentialWrapper

try:
    from torch.distributed.rpc import PyRRef as RpcRef
except ImportError:
    from torch.distributed.rpc import RRef as RpcRef


class ModuleWrapper(nn.Module):
    def __init__(
        self,
        rank: int,
        module: nn.Module,
        module_to_cuda: bool,
        output_to_cpu: bool,
        mem_limit: int,
        slowdown: int,
        timer_config: dict,
        logging_config: dict = None,
        cuda_device: int = -1,
        record_forward_time: bool = False,
    ):
        super(ModuleWrapper, self).__init__()
        # add basic config
        self._rank = rank
        self._module_to_cuda = module_to_cuda
        self._output_to_cpu = output_to_cpu
        self._slowdown = slowdown

        # add module
        assert isinstance(
            module, SequentialWrapper
        ), "The module is of type {}, but expected SequentialWrapper".format(
            type(module)
        )
        self._module = module

        # add memory limit check
        assert (
            mem_limit != 0 and mem_limit >= -1
        ), "mem_limit can only be set to -1 or positive number, if it is -1, it will automatically detect the GPU RAM based on cuda device."
        self._mem_limit = mem_limit

        # logger
        if logging_config:
            self._logger = Logger(**logging_config)
        else:
            self._logger = None

        # timer
        self._timer = DistributedTimer(**timer_config)

        # add slowdown to backward
        if logging_config:
            bwd_logger = Logger(**logging_config)
        else:
            bwd_logger = None

        self._slowdown_module = BackwardSlowdownModule(
            rank=rank,
            slowdown=slowdown,
            timer=self._timer,
            logger=bwd_logger,
            do_slowdown=True,
        )

        self._output_slowdown_module = BackwardSlowdownModule(
            rank=rank,
            slowdown=slowdown,
            timer=self._timer,
            logger=bwd_logger,
            do_slowdown=False,
        )

        # move from cpu to gpu if configured
        assert (
            not module_to_cuda or cuda_device >= 0
        ), "GPU device index must be non-negative"
        if self._module_to_cuda:
            torch.cuda.set_device(cuda_device)
            self.cuda()
        self._gpu_index = cuda_device

        self._record_forward_time = record_forward_time
        if self._record_forward_time:
            self.forward_time = []

    @property
    def rank(self):
        return self._rank

    @property
    def gpu_index(self):
        return self._gpu_index

    def _time_forward(self, *args):
        """
        Time the forward pass and log into file if logger is present. Slow down
        by manually calling time.sleep(n) to simulate different computing power
        """

        # preprocess
        args = self._process_data_before(args)

        # forward
        start = dutils.get_time()
        output = self._module(*args)
        end = dutils.get_time()

        # slowdown
        comp_time = end - start
        if self._slowdown > 0:
            time.sleep(comp_time * self._slowdown)

        # log
        real_end = dutils.get_time()
        if self._logger:
            self._logger.info(
                "forward time on rank {}: {}".format(self._rank, real_end - start)
            )

        if self._record_forward_time:
            self.forward_time.append(real_end - start)

        # post processing
        output = self._process_data_after(output)
        return output

    def _convert_to_tuple(self, data):
        if isinstance(data, tuple) or isinstance(data, list):
            return data
        else:
            return (data,)

    def _process_data_before(self, args):
        # handle rpc ref
        if len(args) == 1 and isinstance(args[0], RpcRef):
            args = self._fetch_data_before(args[0])
        args = [self._move_data_before(arg) for arg in args]
        args = [self._slowdown_backward(arg) for arg in args]
        return args

    def _process_data_after(self, output):
        output = self._convert_to_tuple(output)
        output = [self._slowdown_output_backward(arg) for arg in output]
        output = tuple(output)
        output = [self._move_data_after(data) for data in output]
        return output

    def _fetch_data_before(self, data):
        output = data.to_here()
        return output

    def _move_data_before(self, data):
        if self._module_to_cuda and isinstance(data, torch.Tensor):
            data = data.to("cuda:{}".format(self._gpu_index))
        return data

    def _move_data_after(self, data):
        if self._output_to_cpu and isinstance(data, torch.Tensor):
            data = data.cpu()
        return data

    def _slowdown_backward(self, data):
        if isinstance(data, torch.Tensor):
            data = self._slowdown_module(data)
        return data

    def _slowdown_output_backward(self, data):
        if isinstance(data, torch.Tensor):
            data = self._output_slowdown_module(data)
        return data

    def detect_mem(self, destroy_module: bool):
        """
        If mem_limit is -1, it means automatically detect the RAM.
        If it is a positive number, it means to use this set value as memory limit.

        The RAM should be in MB.
        """
        if destroy_module:
            # delete module to get more accurate memory reading
            del self._module
            if self._module_to_cuda:
                torch.cuda.empty_cache()

        if self._mem_limit > 0:
            return self._mem_limit
        elif self._mem_limit == -1:
            if self._module_to_cuda:
                return self._detect_gpu_ram()
            else:
                return self._detect_cpu_ram()
        else:
            raise ValueError("Invalid value {} for mem_limit".format(self._mem_limit))

    def _detect_gpu_ram(self):
        # get available mem
        _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        # minus 800 to avoid OOM
        mem = memory_free_values[self._gpu_index] - 500
        return mem

    def _detect_cpu_ram(self):
        avai_mem = psutil.virtual_memory().available
        avai_mem = avai_mem / 1024 / 1024
        return avai_mem

    def forward(self, *args):
        """
        Args:
            *args: a tuple containing all the inputs

        Returns: a tuple of tensors

        """
        if isinstance(self, RpcRef):
            self = self.to_here()
        output = self._time_forward(*args)
        return output


class BackwardSlowdownFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, rank, slowdown, timer, logger, do_slowdown):
        ctx.slowdown = slowdown
        ctx.timer = timer
        ctx.logger = logger
        ctx.rank = rank
        ctx.do_slowdown = do_slowdown

        ctx.save_for_backward(feat)
        output = feat.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # slowdown
        timer = ctx.timer
        logger = ctx.logger

        if ctx.do_slowdown:
            dutils.synchronize()
            timer.add_timestamp()

            backward_time = timer.get_prev_interval()

            if ctx.slowdown > 0:
                time.sleep(max(0, backward_time * ctx.slowdown))

            # log
            if logger:
                logger.info(
                    "backward time on rank {}: {}".format(
                        ctx.rank, backward_time * (ctx.slowdown + 1)
                    )
                )

        grad_input = grad_output.clone()

        dutils.synchronize()
        timer.add_timestamp()

        # the number of output of backward should be the same
        # as that of input of forward
        return grad_input, None, None, None, None, None, None


class BackwardSlowdownModule(nn.Module):
    def __init__(self, rank, slowdown, timer, logger, do_slowdown):
        super().__init__()
        self.rank = rank
        self.slowdown = slowdown
        self.timer = timer
        self.logger = logger
        self.do_slowdown = do_slowdown
        self.func = BackwardSlowdownFunction.apply

    def forward(self, data):
        return self.func(
            data, self.rank, self.slowdown, self.timer, self.logger, self.do_slowdown
        )
