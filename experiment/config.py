import math
import os.path as osp
import os
from dllb.model.bert import BertConfig
import numpy as np

# project path
PROJECT = os.getenv("PROJECT")

# allocation type, valid values are optimal, even and dynamic
ALLOCATE_TYPE = "even"

# num of training node (including central server)
CORE_NUM = 4

# num of hidden layers
LAYER_NUM = 10

# SOME PARAMS
__data_root = f"{PROJECT}/BERT/data/glue_data"
__task = "MNLI"
__model_root = f"{PROJECT}/BERT/model/wwm_uncased_L-24_H-1024_A-16"
__config_file = osp.join(__model_root, "bert_config.json")
__config = BertConfig.from_json_file(__config_file)

__ENCODER = [
    dict(layer_type="BertLayer_Head", config=__config.__dict__),
    dict(layer_type="BertLayer_Body", config=__config.__dict__),
    dict(layer_type="BertLayer_Tail", config=__config.__dict__),
] * LAYER_NUM  # __config.num_hidden_layers

__BERT_LAYERS = (
    [
        dict(
            layer_type="BertEmbeddings",
            config=__config.__dict__,
        )
    ]
    + __ENCODER
    + [
        dict(layer_type="BertPooler", config=__config.__dict__),
        dict(
            layer_type="BertTailForClassification",
            hidden_dropout_prob=__config.hidden_dropout_prob,
            hidden_size=__config.hidden_size,
            num_classes=3,
        ),
    ]
)

# config for rpc initialization
# will be replaced in SLURM job script
rpc_config = dict(
    MASTER_ADDR="localhost", MASTER_PORT="29500", GLOO_SOCKET_IFNAME="ipogif0"
)

# for runner logger hook
__LOG_ROOT = f"{PROJECT}/logs/{CORE_NUM}nodes_{LAYER_NUM}layers/{ALLOCATE_TYPE}"
logging_config = dict(
    mode="a", filename=osp.join(__LOG_ROOT, "allocation.log")  # do not change
)

# worker config
worker_config = []


def get_slowdown(val, num):
    # generate reproducible random slowdown
    rng = np.random.default_rng(seed=35)
    rints = rng.integers(low=1, high=7, size=num + 1)
    return rints[val]

WORKER_NUM = CORE_NUM - 1

for i in range(1, WORKER_NUM + 1):
    worker_config.append(
        dict(
            name=f"gpu-{i}",
            server_config=dict(
                host="localhost",
                port="8001",
            ),
            extra_config=dict(
                # slowdown=get_slowdown(i, WORKER_NUM),
                slowdown=1,  # TODO remove this method
                logging_config=dict(
                    mode="a",  # do not change
                    filename=osp.join(__LOG_ROOT, f"node-{i}-train.log"),
                ),
                mem_limit=-1,
                cuda_device=0,
                module_to_cuda=True,
                output_to_cpu=True,
                timer_config=dict(
                    root=__LOG_ROOT,
                ),
            ),
        ),
    )

# model config
model_config = __BERT_LAYERS

# dataset config
data_config = dict(
    dataset_cfg=dict(
        type="GlueDataset",
        data_dir=osp.join(__data_root, __task),
        bert_model="large-uncased",
        vocab_file=osp.join(__model_root, "vocab.txt"),
        max_seq_length=128,
        do_lower_case=False,
        processor="mnli",
    ),
    dataloader_cfg=dict(
        batch_size=32,
        shuffle=True,
        num_workers=0,
    ),
)

# dynamic allocation config
allocator_config = dict(
    type=ALLOCATE_TYPE,
    benchmark_config=dict(
        model=dict(
            device="cpu",
            param_scale=2,
            data_generator_cfg=dict(
                generator_type="DataloaderGenerator", generator_cfg=data_config
            ),
        ),
        device=dict(
            model_config=[
                dict(
                    layer_type="Conv2d",
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                )
            ]
            * 10,
            iterations=30,
            data_generator_cfg=dict(
                generator_type="RandomTensorGenerator",
                generator_cfg=dict(size=(32, 256, 64, 64)),
            ),
        ),
    ),
)

# training config
train_config = dict(
    optim_cfg=dict(optim_type="SGD", lr=0.001),
    loss_cfg=dict(type="CrossEntropyLoss"),
    runner_cfg=dict(
        max_epochs=1,
        max_iters=30,
    ),
    # for runner hook
    hook_config=[
        dict(type="StopHook", root=__LOG_ROOT),
        dict(type="DistributedTimerHelperHook"),
    ],
    timer_config=dict(root=__LOG_ROOT),
)
