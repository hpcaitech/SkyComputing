#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .worker import Worker


class WorkerManager(object):
    def __init__(self):
        self._worker_pool = []

    @property
    def size(self):
        return len(self._worker_pool)

    @property
    def worker_pool(self):
        return self._worker_pool

    def get_by_id(self, id_str: str, allow_not_found: bool = False) -> Worker:
        for worker in self._worker_pool:
            if worker.id == id_str:
                return worker

        if not allow_not_found:
            raise LookupError(
                "Worker with id {} is not found in the worker pool".format(id_str)
            )
        else:
            return None

    def load_worker_pool_from_config(self, config: dict) -> None:
        for i, worker_config in enumerate(config):
            worker = Worker(rank=i + 1, **worker_config)  # rank 0 is reserved for host
            self._worker_pool.append(worker)

    def assign_model_to_worker(self, rank: int, model_config: dict) -> None:
        for worker in self._worker_pool:
            if worker.rank == rank:
                worker.model_config(model_config)
                return

        raise LookupError(
            "Worker with rank {} is not found in the worker pool".format(rank)
        )

    def add_worker(self, worker_id: str, worker_config: dict) -> None:
        rank = len(self._worker_pool) + 1
        worker = Worker(rank=rank, worker_id=worker_id, **worker_config)
        self._worker_pool.append(worker)

    def _allocate_rank(self) -> None:
        for i, worker in enumerate(self._worker_pool):
            worker.rank = i + 1  # rank 0 is reserved for host

    def remove_worker_by_id(self, id_str: str) -> None:
        worker = self.get_by_id(id_str)
        assert not worker.is_running, "Worker {} is still running".format(id_str)

        self._worker_pool.remove(worker)
        self._allocate_rank()

    def reset_rank_by_order(self):
        self._worker_pool.sort(key=lambda x: x.order)
        self._allocate_rank()

    def serialize(self):
        res = []
        for worker in self._worker_pool:
            res.append(worker.serialize())
        return res

    @staticmethod
    def deserialize(data: list):
        worker_manager = WorkerManager()

        for worker_data in data:
            worker = Worker.deserialize(worker_data)
            worker_manager.worker_pool.append(worker)
        return worker_manager
