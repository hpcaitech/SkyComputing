#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import inspect
import os.path as osp
import sys
from importlib.machinery import SourceFileLoader


class Config(dict):
    """
    Wrap a dictionary object so that we can access
    values as attributes
    """

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        value = super(Config, self).__getitem__(name)
        return value

    def __setattr__(self, name, value):
        super(Config, self).__setitem__(name, value)

    def update(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)
        return self

    @staticmethod
    def from_dict(data: dict):
        config = Config()

        for k, v in data.items():
            config.__setattr__(k, v)
        return config


def _py2dict(py_path: str):
    # pylint: disable=no-value-for-parameter
    """
    Read python file as python dictionary
    """

    assert py_path.endswith(".py")

    py_path = osp.abspath(py_path)
    parent_dir = osp.dirname(py_path)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    module_name = osp.splitext(osp.basename(py_path))[0]
    source_file = SourceFileLoader(fullname=module_name, path=py_path)
    module = source_file.load_module()
    sys.path.pop(0)
    doc = {
        k: v
        for k, v in module.__dict__.items()
        if not k.startswith("__") and not inspect.ismodule(v) and not inspect.isclass(v)
    }
    del sys.modules[module_name]
    return doc


def load_config(file_path: str):
    config_dict = _py2dict(file_path)
    config = Config(config_dict)

    base = config.pop("base", None)

    if base:
        base_config_path = osp.join(osp.dirname(file_path), base)
        base_config_dict = _py2dict(base_config_path)
        base_config = Config(base_config_dict)
        config = base_config.update(config)

    return config
