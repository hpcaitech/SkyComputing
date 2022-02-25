#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import os.path as osp

from dllb.registry import HOOKS

from ..hooks import Hook


@HOOKS.register_module
class CheckpointHook(Hook):
    def __init__(
        self,
        load_checkpoint_from: str = None,
        save_path: str = None,
        save_interval: int = None,
    ):
        self._load_checkpoint_from = load_checkpoint_from
        self._save_interval = save_interval
        self._save_path = save_path

    def before_run(self, runner):
        if self._load_checkpoint_from:
            runner.parameter_server.load_weights_from_file(self._load_checkpoint_from)
            cur_layer = 0
            rpc_model = runner.model

            for idx, module in enumerate(rpc_model.model):
                # get the number of layers on this worker
                num_layers = len(runner.worker_manager.worker_pool[idx].model_config)
                state_dict_list = []

                # get the state dict from parameter server
                for layer_idx in range(cur_layer, cur_layer + num_layers):
                    state_dict = runner.parameter_server.get_state_dict(layer_idx)
                    state_dict_list.append(state_dict)
                cur_layer += num_layers

                # load the weights onto to the module
                module.load_weights(state_dict_list)

    def after_epoch(self, runner):
        if not osp.exists(self._save_path):
            os.mkdir(self._save_path)

        if self.every_n_epochs(runner, self._save_interval):
            # gather weights from workers
            rpc_model = runner.model
            all_state_dict = []

            for module in rpc_model.model:
                state_dict = module.get_state_dict()
                all_state_dict.extend(state_dict)

            # update the weights in parameter server
            for i in range(len(all_state_dict)):
                try:
                    runner.parameter_server.update_weights(all_state_dict[i], i)
                except:
                    raise Exception(
                        "have {} state dicts, have {} layers, error occurs at {}".format(
                            len(all_state_dict),
                            len(runner.parameter_server.module_list),
                            i,
                        )
                    )

            # save weights
            epoch = runner.epoch
            file_name = osp.join(self._save_path, "epoch_{}.pth".format(epoch))
            runner.parameter_server.save_weights_to_file(file_name)
