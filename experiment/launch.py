#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import os.path as osp
import time

import scaelum.utils as dutils
import torch.distributed.rpc as rpc
from scaelum import build_hook, Runner, RpcModel
from scaelum.builder import build_dataloader_from_cfg, build_data_generator
from scaelum.config import load_config
from scaelum.dynamics import Allocator, WorkerManager, ParameterServer, ModelBenchmarker, DeviceBenchmarker
from scaelum.logger import Logger
from torch import optim
from torch.distributed.optim import DistributedOptimizer


def run_process(rank: int,
                world_size: int,
                rpc_config: dict,
                model_config: list = None,
                data_config: dict = None,
                logging_config: dict = None,
                allocator_config: dict = None,
                train_config: dict = None,
                worker_config: list = None,
                ):
    # set env var for rpc init
    for k, v in rpc_config.items():
        os.environ[k] = str(v)

    # init rpc
    print('starting to initialize rpc on rank: {}'.format(rank))
    worker_name = dutils.generate_worker_name(rank)
    rpc.init_rpc(
        name=worker_name,
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=8,
            rpc_timeout=1200  # 600 second timeout
        )
    )
    print('rpc initialized on rank: {}'.format(rank))

    if rank == 0:
        # clean previous logs
        log_workspace = osp.dirname(logging_config['filename'])

        if osp.exists(log_workspace):
            for f in os.listdir(log_workspace):
                os.remove(osp.join(log_workspace, f))
        else:
            os.makedirs(log_workspace)

        # init logging
        logger = Logger(**logging_config)
        logger.info('logger initialized')

        # init worker manager
        worker_manager = WorkerManager()
        worker_manager.load_worker_pool_from_config(worker_config)

        # create parameter server
        parameter_server = ParameterServer(model_config)

        # create dataloder
        dataloader_cfg = data_config['dataloader_cfg'].copy()
        dataset_cfg = data_config['dataset_cfg'].copy()
        data_loader = build_dataloader_from_cfg(
            dataloader_cfg=dataloader_cfg,
            dataset_cfg=dataset_cfg)
        logger.info('created data loader')

        # benchmarking
        benchmark_cfg = allocator_config['benchmark_config'].copy()

        # build model benchmarker
        data_cfg_for_model = benchmark_cfg['model'].pop('data_generator_cfg')
        generator_type = data_cfg_for_model.pop('generator_type')
        data_generator_for_model = build_data_generator(
            module_name=generator_type,
            **data_cfg_for_model
        )

        model_benchmarker = ModelBenchmarker(
            model_config=model_config,
            data_generator=data_generator_for_model,
            **benchmark_cfg['model']
        )

        # build device benchmarker
        data_cfg_for_device = benchmark_cfg['device'].pop('data_generator_cfg')
        generator_type = data_cfg_for_device.pop('generator_type')
        data_generator_for_device = build_data_generator(
            module_name=generator_type,
            **data_cfg_for_device
        )

        device_benchmarker = DeviceBenchmarker(
            worker_manager=worker_manager,
            data_generator=data_generator_for_device,
            **benchmark_cfg['device']
        )

        # build allocator
        allocator = Allocator(
            model_cfg=model_config,
            worker_manager=worker_manager,
            model_benchmarker=model_benchmarker,
            device_benchmarker=device_benchmarker,
        )

        allocation_success = True

        if allocator_config['type'] == "dynamic":
            try:
                worker_manager = allocator.dynamic_allocate()
                logger.info('dynamically allocated model layers based on benchmarking')

                del allocator, device_benchmarker, model_benchmarker, data_generator_for_model, data_generator_for_device
            except Exception:
                allocation_success = False
        elif allocator_config['type'] == "optimal":
            print("using optimal allocate")
            try:
                worker_manager = allocator.optimal_allocate()
                logger.info("use optimal strategy to allocate model layers, may take long time")
                del allocator, device_benchmarker, model_benchmarker, data_generator_for_model, data_generator_for_device
            except Exception:
                allocation_success = False
        else:
            worker_manager = allocator.even_allocate()
            logger.info('Evenly allocated model layers')
            del allocator

        # log worker workload
        for worker in worker_manager.worker_pool:
            logger.info('rank: {}, number of layers: {}'.format(
                worker.rank, len(worker.model_config)))

        if allocation_success:
            # create model
            model = RpcModel(worker_manager=worker_manager)
            logger.info('created model')

            # create optimizer
            # Build DistributedOptimizer.
            optim_mod = getattr(optim, train_config['optim_cfg'].pop('optim_type'))
            dist_optim = DistributedOptimizer(
                optim_mod, model.parameter_rrefs(),
                **train_config['optim_cfg'])
            logger.info('created distrubted optimizer')

            # build runner
            loss_cfg = train_config['loss_cfg'].copy()
            timer_cfg = train_config['timer_config'].copy()
            logging_cfg = logging_config.copy()

            runner = Runner(
                model=model,
                parameter_server=parameter_server,
                worker_manager=worker_manager,
                optimizer=dist_optim,
                loss_cfg=loss_cfg,
                timer_cfg=timer_cfg,
                logging_cfg=logging_cfg,
                **train_config['runner_cfg']
            )
            logger.info('created runner')

            for cfg in train_config['hook_config']:
                cfg_copy = cfg.copy()
                hook_name = cfg_copy.pop('type')
                hook = build_hook(hook_name, **cfg_copy)
                runner.register_hook(hook)
            logger.info('register hooks')

            runner.train(data_loader)
    else:
        time.sleep(30)
    rpc.shutdown()
    print('finish')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='path to config file')
    # parser.add_argument('-n', '--ntasks', type=int)
    # parser.add_argument('-t', '--type', type=str)
    parser.add_argument('-p', '--port', type=int, default=29500)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    # get individual config
    model_config = config.pop('model_config')
    rpc_config = config.pop('rpc_config')
    data_config = config.pop('data_config')
    logging_config = config.pop('logging_config')
    worker_config = config.pop('worker_config')
    allocator_config = config.pop('allocator_config')
    train_config = config.pop('train_config')

    # replace the rpc init environment variable
    host_file = "./HOST"
    with open(host_file, 'r') as f:
        host = f.readline().strip()

    rpc_config['MASTER_ADDR'] = host
    rpc_config['MASTER_PORT'] = args.port

    # set Language to avoid error on Frontera
    os.environ['LC_ALL'] = 'C.UTF-8'

    # get ibrun rank and world size
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])

    run_process(
        rank=rank,
        world_size=world_size,
        rpc_config=rpc_config,
        model_config=model_config,
        data_config=data_config,
        logging_config=logging_config,
        allocator_config=allocator_config,
        train_config=train_config,
        worker_config=worker_config,
    )