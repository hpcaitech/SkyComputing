# Sky Computing

## Introduction

Sky Computing is a load-balanced framework for federated learning model parallelism. It adaptively allocate model layers to devices based on the their hardware sepcification. Sky Computing outperforms the baseline method by 55% in training time when training 160-layer BERT in a 64-node cluster. Our paper can be found at https://arxiv.org/abs/2202.11836

## Installation

```shell
git clone git@github.com:hpcaitech/SkyComputing.git
python -m pip install -r requirements.txt
cd ./scaelum
python -m pip install -v -e .
```

## Experiment (using BERT)

To benchmark the Sky Computing, we prepared a single demo which you can run on your cluster to train BERT.

### Prepare BERT model

Bidirectional Encoder Representations from Transformers (aka [BERT](https://aclanthology.org/N19-1423/)) is one of the state-of-the-art deep learning models for Natural Language Processing. In the experiment part, we use BERT to run a simple benchmark.

```shell
cd $PROJECT
mkdir -p BERT/model && cd BERT/model 
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip
```

### Prepare GLUE MNLI dataset

The General Language Understanding Evaluation (aka [GLUE](https://gluebenchmark.com/)) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. And the Multi-Genre Natural Language Inference (aka [MNLI](https://cims.nyu.edu/~sbowman/multinli/)) is one of the tasks in GLUE, it is a crowd-sourced collection of 433k sentence pairs annotated with textual entailment information.

```shell
cd $PROJECT
mkdir -p BERT/data && cd BERT/data
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/1502038877f6a88c225a34450793fbc3ea87eaba/download_glue_data.py
python download_glue_data.py --data_dir ./glue_data --tasks MNLI
```

### Configuration

To run dllb in your cluster, you need to write a config file which contains the necessary information about training, e.g. model layers, useful environment variables. We have provided a well-commentted [example](https://github.com/hpcaitech/SkyComputing/blob/main/experiment/config.py), and here are some most important option:

```python
# your project path
PROJECT = os.getenv("PROJECT")

# allocation type, valid values are even, optimal and dynamic
ALLOCATE_TYPE = "even"

# num of node (including the central server)
CORE_NUM = 4
```

### Run scripts

[Slurm](https://www.schedmd.com/) is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. We used slurm script to run our experiment.

```shell
#!/bin/sh

#SBATCH --job-name=gpu16   # Job name
#SBATCH -o gpu16.o%j       # Name of stdout output file
#SBATCH -e gpu16.e%j       # Name of stderr error file
#SBATCH -N 16              # Node numbers
#SBATCH -n 16              # GPU numbers
#SBATCH --time=02:00:00    # Run time (hh:mm:ss)

# run
python ./ip_addr.py > "./HOST"
srun python ./launch.py -c "./experiment/config.py"
```

## Citation

```tex
@misc{zhu2022sky,
      title={Sky Computing: Accelerating Geo-distributed Computing in Federated Learning}, 
      author={Jie Zhu and Shenggui Li and Yang You},
      year={2022},
      eprint={2202.11836},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
