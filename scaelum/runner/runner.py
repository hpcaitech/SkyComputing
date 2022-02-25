#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from scaelum import utils as dutils, WorkerManager
from scaelum.dynamics import ParameterServer
from scaelum.logger import Logger
from scaelum.runner import Hook
from scaelum.timer import DistributedTimer
from torch.distributed.optim import DistributedOptimizer


class Runner:
    def __init__(
        self,
        model: nn.Module,
        parameter_server: ParameterServer,
        worker_manager: WorkerManager,
        optimizer: DistributedOptimizer,
        max_epochs: int,
        max_iters: int,
        loss_cfg: dict,
        timer_cfg: dict,
        logging_cfg: dict,
    ):
        # model and optimizer instance
        self.model = model
        self.worker_manager = worker_manager
        self.parameter_server = parameter_server
        self.optimizer = optimizer

        # param
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epoch = max_epochs
        self._max_iter = max_iters

        # logger
        self._logging_config = logging_cfg
        self._logger = Logger(**logging_cfg)

        # timer
        self._timer_config = timer_cfg
        self._timer = DistributedTimer(**timer_cfg)

        # build loss
        loss_name = loss_cfg.pop("type")
        self.loss_function = getattr(nn, loss_name)(**loss_cfg)

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @epoch.setter
    def epoch(self, i):
        self._epoch = i

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @iter.setter
    def iter(self, i):
        self._iter = i

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iter(self):
        """int: Maximum training iterations."""
        return self._max_iter

    def register_hook(self, hook):
        assert isinstance(hook, Hook)
        self._hooks.append(hook)

    def _call_hook(self, fn_name):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def train(self, data_loader):
        # set model to train
        self.model.train(True)
        self._call_hook("before_run")

        # train by epoch
        while self.epoch < self._max_epoch:
            # call hook func
            self._call_hook("before_train_epoch")

            # train by iter
            for batch_index, (data, labels) in enumerate(data_loader):

                # break if max iter is exceeded
                if self._iter > self._max_iter:
                    break

                self._logger.info("epoch: {}, iter: {}".format(self.epoch, self.iter))

                # call hook func
                self._call_hook("before_train_iter")

                with dist_autograd.context() as context_id:
                    # forward
                    fwd_start = dutils.get_time()
                    outputs = self.model(data)
                    loss = self.loss_function(outputs, labels)
                    fwd_end = dutils.get_time()

                    # Backward pass (run distributed autograd).
                    bwd_start = dutils.get_time()
                    self._timer.add_timestamp()
                    dist_autograd.backward(context_id, [loss])
                    bwd_end = dutils.get_time()
                    self.optimizer.step(context_id)
                    step_end = dutils.get_time()

                    # log time
                    self._logger.info("forward time: {}".format(fwd_end - fwd_start))
                    self._logger.info("backward time: {}".format(bwd_end - bwd_start))
                    self._logger.info("step time: {}".format(step_end - bwd_end))

                # update iter
                self._iter += 1
                self._call_hook("after_train_iter")

            # update epoch
            self._epoch += 1
            self._call_hook("after_train_epoch")

        # finish training
        self._call_hook("after_run")
