#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import os.path as osp

from scaelum.registry import HOOKS

from ..hooks import Hook


@HOOKS.register_module
class StopHook(Hook):
    def __init__(self, root="/tmp"):
        super().__init__()
        self.file_path = osp.join(root, "stop_flag.txt")

    def after_iter(self, runner):
        with open(self.file_path, "r") as f:
            flag = f.readline().strip()

            if flag == "1":
                runner.iter = runner.max_iters + 1
                runner.epoch = runner.max_epochs + 1

    def before_run(self, runner):
        with open(self.file_path, "w") as f:
            f.write("0")

    def after_run(self, runner):
        if osp.exists(self.file_path):
            os.remove(self.file_path)

    @staticmethod
    def stop(root):
        file_path = osp.join(root, "stop_flag.txt")
        with open(file_path, "w") as f:
            f.write("1")
