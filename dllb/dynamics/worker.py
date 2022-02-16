#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import uuid


class Worker(object):
    def __init__(
        self,
        rank: int,
        name: int,
        server_config: dict,
        worker_id: str = None,
        order: int = None,
        model_config: list = None,
        extra_config: dict = None,
        is_running: bool = False,
    ) -> None:

        self._rank = rank
        self._name = name
        self._is_running = is_running
        self._order = order
        if worker_id is None:
            self._worker_id = uuid.uuid4().__str__()
        else:
            self._worker_id = worker_id

        # configs
        self._server_config = server_config
        self._model_config = model_config
        self._extra_config = extra_config

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def id(self) -> str:
        return self._worker_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_config(self) -> list:
        return self._model_config

    @property
    def env_config(self) -> dict:
        return self._env_config

    @property
    def server_config(self) -> dict:
        return self._server_config

    @property
    def extra_config(self) -> dict:
        return self._extra_config

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def order(self) -> None:
        return self._order

    @order.setter
    def order(self, ord: int) -> None:
        self._order = ord

    @is_running.setter
    def is_running(self, status: bool) -> None:
        self._is_running = status

    @model_config.setter
    def model_config(self, config: list) -> None:
        self._model_config = config

    @rank.setter
    def rank(self, rank) -> None:
        self._rank = rank

    def serialize(self):
        return self.__dict__

    @staticmethod
    def deserialize(data: dict):
        kwargs = dict()

        for k, v in data.items():
            kwargs[k.lstrip("_")] = v

        return Worker(**kwargs)
