#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from dllb.registry import HOOKS

from ..hooks import Hook


@HOOKS.register_module
class DistributedTimerHelperHook(Hook):
    def before_run(self, runner):
        runner._timer.clean_prev_file()

    def after_run(self, runner):
        runner._timer.clean_prev_file()
