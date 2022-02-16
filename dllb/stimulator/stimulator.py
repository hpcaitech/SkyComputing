import numpy as np


class Stimulator(object):
    def __init__(self, worker_num: int):
        self.worker_num = worker_num
        # generate random slowdown [1, 3) for memory usage
        m_rng = np.random.default_rng(seed=22)
        self.m_slowdown = 2 * m_rng.random((worker_num + 1,)) + 1
        # generate random slowdown [1, 2) for network usage
        n_rng = np.random.default_rng(seed=32)
        self.n_slowdown = n_rng.random((worker_num + 1,)) + 1
        # generate random slowdown [1, 4) for computing power
        c_rng = np.random.default_rng(seed=32)
        self.c_slowdown = c_rng.random((worker_num + 1,)) + 1

    def memory_slowdown(self, worker_id: int) -> float:
        return self.m_slowdown[worker_id]

    def compute_slowdown(self, worker_id: int) -> float:
        return self.c_slowdown[worker_id]

    def network_stimulate(self, worker_id: int) -> float:
        return self.n_slowdown[worker_id]
