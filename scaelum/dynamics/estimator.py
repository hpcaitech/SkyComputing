#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from dllb import utils
from pthflops import count_ops


class Estimator:
    @staticmethod
    def benchmark_speed(model, data, device, iterations, dtype=None):
        # convert to tuple
        data = Estimator._convert_to_tuple(data)

        # convert data type
        if dtype:
            data = Estimator._convert_dtype(data, dtype)

        # move to device
        data = Estimator._move_to_device(data, device)
        model = model.to(device)

        # measure forward time
        with torch.no_grad():
            for i in range(iterations):
                model(*data)
        end = utils.get_time()
        total_time = sum(model.forward_time)
        # print('forward time on rank {}: {}'.format(model.rank, model.forward_time))
        return total_time

    @staticmethod
    def _convert_dtype(data, dtype):
        assert isinstance(dtype, str)
        dtype = getattr(torch, dtype)
        data = [_data.to(dtype) for _data in data]
        return data

    @staticmethod
    def _move_to_device(data, device):
        data = [_data.to(device) for _data in data]
        return data

    @staticmethod
    def _convert_to_tuple(data):
        if isinstance(data, (list, tuple)):
            return data
        else:
            return (data,)

    @staticmethod
    def benchmark_model(model, data, device, dtype=None, param_scale=2):
        # convert to tuple
        data = Estimator._convert_to_tuple(data)

        # convert dtype
        if dtype:
            data = Estimator._convert_dtype(data, dtype)

        # move to device
        data = Estimator._move_to_device(data, device)
        model = model.to(device)

        # get output
        output = model(*data)

        flops = Estimator._calc_flops(model, data)
        mem_usage = Estimator._calc_memory_usage(model, data, param_scale)
        return output, flops, mem_usage

    @staticmethod
    def _calc_flops(model, data):
        # need to convert to tuple for multiple inputs for jit tracing
        if isinstance(data, list):
            data = tuple(data)
        assert isinstance(data, tuple)
        flops, _ = count_ops(model, data, print_readable=False)
        return flops

    @staticmethod
    def _calc_memory_usage(model, data, param_scale):
        assert isinstance(data, (list, tuple))

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [list(o.size()) for o in output]
                else:
                    summary[m_key]["output_shape"] = [list(output.size())]

                # TODO: add reserved memory for CUDA and use a.nelement() * a.element_size() to calculate size
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if not isinstance(module, nn.Sequential) and not isinstance(
                module, nn.ModuleList
            ):
                hooks.append(module.register_forward_hook(hook))

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        model(*data)

        # remove these hooks
        for h in hooks:
            h.remove()

        total_params = 0
        total_output = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            total_params += summary[layer]["nb_params"]
            total_output += sum(
                [
                    np.prod(output_shape)
                    for output_shape in summary[layer]["output_shape"]
                ]
            )

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(
            sum([np.prod(_data.size()) * 4.0 / (1024**2.0) for _data in data])
        )
        total_output_size = abs(
            2.0 * total_output * 4.0 / (1024**2.0)
        )  # x2 for gradients
        total_params_size = abs(
            param_scale * total_params * 4.0 / (1024**2.0)
        )  # x param_scale for backward
        total_size = total_params_size + total_output_size + total_input_size

        # return summary
        return total_size.item()
