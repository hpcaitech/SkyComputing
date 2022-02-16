#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from loguru import logger
import pulp

from .benchmarker import ModelBenchmarker, DeviceBenchmarker
from .worker_manager import WorkerManager


class Allocator(object):
    def __init__(
        self,
        model_cfg: dict,
        worker_manager: WorkerManager,
        model_benchmarker: ModelBenchmarker,
        device_benchmarker: DeviceBenchmarker,
    ):
        self._model_cfg = model_cfg
        self._worker_manager = worker_manager
        self._model_benchmarker = model_benchmarker
        self._device_benchmarker = device_benchmarker

    def optimal_allocate(self, max_time=300, threads=24):

        # benchmark
        worker_ranks, workers_performance = zip(
            *self._device_benchmarker.benchmark().items()
        )
        lf, lm = self._model_benchmarker.benchmark()
        # logger.info(f"layers flops: {lf}")
        # logger.info(f"layers memories: {lm}")

        D = len(worker_ranks)
        L = len(lf)

        # parse the results
        worker_ranks = [int(item.lstrip("worker")) for item in worker_ranks]
        logger.info(f"worker ranks: {worker_ranks}")
        dt = [item["time"] for item in workers_performance]
        logger.info(f"worker time: {dt}")
        dm = [item["avai_mem"] for item in workers_performance]
        # logger.info(f"worker memory limit: {dm}")

        # solve problem
        model = pulp.LpProblem("optimal_allocate", pulp.LpMinimize)
        logger.info("set up MIP")

        # create variables
        x = pulp.LpVariable.matrix("x", (range(D), range(L)), cat=pulp.LpBinary)
        y = pulp.LpVariable.matrix("y", range(D), lowBound=0, upBound=L)
        z = pulp.LpVariable.matrix("z", range(D), lowBound=0, upBound=L)
        q = pulp.LpVariable("max_device_time")
        logger.info("added all variables")

        # add one feasible solution to pre-solve
        avg_num_layer = math.floor(L / D)
        num_remain_layer = L - avg_num_layer * D

        bias = 0
        for row, device in enumerate(x):
            start_idx = row * avg_num_layer + bias
            if num_remain_layer > 0:
                num_remain_layer -= 1
                bias += 1
            end_idx = row * avg_num_layer + avg_num_layer - 1 + bias
            z[row].setInitialValue(start_idx)
            y[row].setInitialValue(end_idx)
            for col, layer in enumerate(device):
                if start_idx <= col <= end_idx:
                    layer.setInitialValue(1)
                else:
                    layer.setInitialValue(0)
        logger.info("added one feasible solution")

        # objective function
        model.objective = q
        logger.info("add obj.")

        # add constraints

        # constraint 1
        for i in range(D):
            model += (
                pulp.LpAffineExpression([(x[i][j], lm[j]) for j in range(L)]) <= dm[i]
            )

        # constraint 2 and 3
        for i in range(D):
            for j in range(L):
                model += y[i] >= j * x[i][j]
                model += z[i] <= j * x[i][j] + (L + 1) * (1 - x[i][j])

        # constraint 4
        for i in range(D):
            model += y[i] - z[i] <= pulp.lpSum(x[i][j] for j in range(L)) - 1

        # constraint 5
        for j in range(L):
            model += pulp.lpSum(x[i][j] for i in range(D)) == 1

        # constraint 6
        for i in range(D):
            model += q >= dt[i] * pulp.lpSum(x[i][j] * lf[j] for j in range(L))

        logger.info("added all constraints")

        solver_list = pulp.listSolvers(onlyAvailable=True)

        if "GUROBI_CMD" in solver_list:
            logger.info("using gurobi as solver")
            model.solve(
                pulp.GUROBI_CMD(
                    timeLimit=max_time,
                    msg=True,
                    gapRel=0.2,
                    threads=threads,
                    warmStart=True,
                )
            )
        else:
            logger.info("using CBC as solver")
            model.solve(
                pulp.PULP_CBC_CMD(
                    timeLimit=max_time,
                    msg=True,
                    gapRel=0.2,
                    threads=threads,
                    warmStart=True,
                )
            )

        for i in z:
            print(i.value(), end=" ")
        print()
        for i in y:
            print(i.value(), end=" ")
        print()

        # allocate to
        partition = []
        for i in range(D):
            info = {
                "rank": worker_ranks[i],
                "start": int(z[i].value()),
                "end": int(y[i].value()),
            }
            partition.append(info)
        # sort partition by idx
        partition.sort(key=lambda t: t["start"])
        print(partition)

        for i, info in enumerate(partition):
            for worker in self._worker_manager.worker_pool:
                if info["rank"] == worker.rank:
                    print(f"rank {worker.rank}", end=" ")
                    layers = self._model_cfg[info["start"] : info["end"] + 1]
                    print(f"has layer {info['start']} to layer {info['end']}", end=" ")
                    worker.model_config = layers
                    print("and set up new config")
                    worker.order = i + 1
                    print(f"rank {worker.rank}'s order: {worker.order}")

        # for i, rank in enumerate(worker_ranks):
        #     for worker in self._worker_manager.worker_pool:
        #         if worker.rank == rank:
        #             layers = self._model_cfg[int(z[i].value()):int(y[i].value())]
        #             print(f"rank {rank} has layer {int(z[i].value())} to {int(y[i].value())}")
        #             worker.model_config = layers
        #             worker.order = i + 1

        self._worker_manager.reset_rank_by_order()
        print("reset by order")

        for worker in self._worker_manager.worker_pool:
            print(worker.rank)

        return self._worker_manager

    def dynamic_allocate(self, break_iter=1000):
        """
        Allocate the layers dynamically among the workers
        """

        # get results
        worker_time_and_avai_mem = self._device_benchmarker.benchmark()
        layer_flops, layer_mem = self._model_benchmarker.benchmark()

        print("worker_time_and_avai_mem: {}".format(worker_time_and_avai_mem))
        # print('layer_flops: {}'.format(layer_flops))
        # print('layer_mem: {}'.format(layer_mem))

        # parse the results
        worker_time_and_avai_mem = list(worker_time_and_avai_mem.items())
        worker_ranks = [
            int(item[0].lstrip("worker")) for item in worker_time_and_avai_mem
        ]
        worker_time = [item[1]["time"] for item in worker_time_and_avai_mem]
        worker_avai_mem = [item[1]["avai_mem"] for item in worker_time_and_avai_mem]

        # check if the smallest worker avai mem can hold the smallest layer
        assert min(worker_avai_mem) > min(
            layer_mem
        ), "The smallest worker has insufficient memory for smallest layer"

        # create partition index
        num_layer = len(layer_flops)
        num_worker = len(worker_ranks)
        avg_num_layers = math.floor(num_layer / num_worker)
        remainder = num_layer - avg_num_layers * num_worker
        num_layers_on_worker = [avg_num_layers] * num_worker

        for i in range(num_worker):
            if remainder > 0:
                num_layers_on_worker[i] += 1
                remainder -= 1
            else:
                break
        partition_idx = [0] + [
            sum(num_layers_on_worker[:idx]) for idx in range(1, num_worker + 1)
        ]

        # partition based on benchmark results
        partition_idx = self._allocate_by_mem(
            worker_rank=worker_ranks,
            partition_idx=partition_idx,
            worker_avai_mem=worker_avai_mem,
            layer_mem=layer_mem,
        )
        partition_idx = self._allocate_by_flops_time(
            worker_rank=worker_ranks,
            partition_idx=partition_idx,
            worker_time=worker_time,
            layer_flops=layer_flops,
            worker_avai_mem=worker_avai_mem,
            layer_mem=layer_mem,
            break_iter=break_iter,
        )

        # allocate to configs
        for i, rank in enumerate(worker_ranks):
            for worker in self._worker_manager.worker_pool:
                if worker.rank == rank:
                    print(f"rank {worker.rank}", end=" ")
                    layers = self._model_cfg[partition_idx[i] : partition_idx[i + 1]]
                    print(
                        f"rank {rank} has layer {int(partition_idx[i])} to {partition_idx[i + 1]}"
                    )
                    worker.model_config = layers
                    worker.order = i + 1

        self._worker_manager.reset_rank_by_order()
        for worker in self._worker_manager.worker_pool:
            print(worker.rank, end=" ")

        return self._worker_manager

    def even_allocate(self):
        """
        Allocate the layers equally among the workers based on the number of layers
        """
        num_worker = len(self._worker_manager.worker_pool)
        num_layer = len(self._model_cfg)
        avg_num_layer = math.floor(num_layer / num_worker)
        num_remain_layer = num_layer - avg_num_layer * num_worker
        cur_layer_idx = 0

        for idx, worker in enumerate(self._worker_manager.worker_pool):
            if num_remain_layer > 0:
                num_remain_layer -= 1
                cur_num_layer = avg_num_layer + 1
            else:
                cur_num_layer = avg_num_layer

            layers = self._model_cfg[cur_layer_idx : cur_layer_idx + cur_num_layer]
            worker.model_config = layers
            cur_layer_idx = cur_layer_idx + cur_num_layer

        return self._worker_manager

    def _get_num_layers_on_worker(self, index, partition_idx):
        return partition_idx[index + 1] - partition_idx[index]

    def _is_last_worker(self, index, worker_rank):
        return index == len(worker_rank) - 1

    def _list_greater_than(self, l1, l2):
        for x, y in zip(l1, l2):
            if x < y:
                return False

        return True

    def _allocate_by_flops_time(
        self,
        worker_rank,
        partition_idx,
        worker_time,
        layer_flops,
        worker_avai_mem,
        layer_mem,
        break_iter,
    ):
        # normalize time results
        worker_time = [item / min(worker_time) for item in worker_time]

        # iteratively update partition index based on flops * time
        iter = 0
        while True:
            # calculate flops on each worker
            workers_flops_time_allocated = [
                sum(layer_flops[partition_idx[j] : partition_idx[j + 1]])
                * worker_time[j]
                for j in range(len(worker_rank))
            ]

            # set the target flops * time on average
            target = sum(workers_flops_time_allocated) // len(worker_rank)

            old_partition_idx = partition_idx[:]

            for j in range(len(worker_rank) - 1):
                current_workload = (
                    sum(layer_flops[partition_idx[j] : partition_idx[j + 1]])
                    * worker_time[j]
                )

                if (
                    current_workload < target
                    and self._get_num_layers_on_worker(j + 1, partition_idx) > 1
                ):
                    # add a layer if memory allows
                    expected_ram_allocated = sum(
                        layer_mem[partition_idx[j] : partition_idx[j + 1] + 1]
                    )
                    if expected_ram_allocated < worker_avai_mem[j]:
                        partition_idx[j + 1] += 1
                else:
                    last_layer_workload_on_this_device = (
                        layer_flops[partition_idx[j + 1] - 1] * worker_time[j]
                    )
                    workload_on_next_device = (
                        sum(layer_flops[partition_idx[j] : partition_idx[j + 1]])
                        * worker_time[j]
                    )

                    if (
                        workload_on_next_device < target
                        and current_workload
                        > target + last_layer_workload_on_this_device
                        and self._get_num_layers_on_worker(j, partition_idx) > 1
                    ):
                        next_worker_expected_ram_allocated = sum(
                            layer_mem[partition_idx[j + 1] - 1 : partition_idx[j + 2]]
                        )
                        if next_worker_expected_ram_allocated < worker_avai_mem[j + 1]:
                            partition_idx[j + 1] -= 1

            if old_partition_idx == partition_idx:
                break

            iter += 1

            if iter == break_iter:
                break

        return partition_idx

    def _allocate_by_mem(self, worker_rank, partition_idx, worker_avai_mem, layer_mem):
        # flag for if allocation satisfy memory requirement
        mem_satisfy = False

        def _compute_mem_allocated(lm, pi, wr):
            return [sum(lm[pi[j] : pi[j + 1]]) for j in range(len(wr))]

        # iteratively update partition index based on mem_avai and mem_allocated
        while True:
            # calculate flops on each worker
            workers_mem_allocated = _compute_mem_allocated(
                layer_mem, partition_idx, worker_avai_mem
            )

            # break the loop if mem allocated < avai mem on each worker
            if self._list_greater_than(worker_avai_mem, workers_mem_allocated):
                mem_satisfy = True
                break

            old_partition_idx = partition_idx[:]

            for j in range(len(worker_rank) - 1):
                while (
                    workers_mem_allocated[j] > worker_avai_mem[j]
                    and partition_idx[j + 1] - partition_idx[j] > 1
                ):
                    # remove a layer if memory is not enough
                    partition_idx[j + 1] -= 1
                    workers_mem_allocated = _compute_mem_allocated(
                        layer_mem, partition_idx, worker_avai_mem
                    )

                    if self._list_greater_than(worker_avai_mem, workers_mem_allocated):
                        mem_satisfy = True
                        break

                if mem_satisfy:
                    break

                # add a layer if memory allows
                while (
                    workers_mem_allocated[j] < worker_avai_mem[j]
                    and partition_idx[j + 2] - partition_idx[j + 1] > 1
                ):
                    expected_ram_allocated = sum(
                        layer_mem[partition_idx[j] : partition_idx[j + 1] + 1]
                    )
                    if expected_ram_allocated < worker_avai_mem[j]:
                        partition_idx[j + 1] += 1
                        workers_mem_allocated = _compute_mem_allocated(
                            layer_mem, partition_idx, worker_avai_mem
                        )
                    else:
                        break

                    if self._list_greater_than(worker_avai_mem, workers_mem_allocated):
                        mem_satisfy = True
                        break

                if mem_satisfy:
                    break

            if old_partition_idx == partition_idx:
                break

        if mem_satisfy:
            return partition_idx
        else:
            print(partition_idx)
            raise Exception("memory allocation failed")
