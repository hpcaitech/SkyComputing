#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os.path as osp
import pickle

import torch
from dllb.registry import DATASET
from torch.utils.data import Dataset, TensorDataset

from .glue.processor import PROCESSORS, convert_examples_to_features
from .glue.tokenization import BertTokenizer


@DATASET.register_module
class GlueDataset(Dataset):
    def __init__(
        self, data_dir, bert_model, vocab_file, max_seq_length, do_lower_case, processor
    ):
        super().__init__()
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.tokenizer = BertTokenizer(
            vocab_file,
            do_lower_case=do_lower_case,
            max_len=512,
        )
        self.processor = PROCESSORS[processor]()
        self.dataset = self._build_dataset()

    def __getitem__(self, idx):
        items = self.dataset.__getitem__(idx)

        return items[:3], items[-1]

    def __len__(self):
        return self.dataset.__len__()

    def _get_train_features(self):
        cached_train_features_file = osp.join(
            self.data_dir,
            "{0}_{1}_{2}".format(
                list(filter(None, self.bert_model.split("/"))).pop(),
                str(self.max_seq_length),
                str(self.do_lower_case),
            ),
        )
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            print("Converting examples to features")
            train_examples = self.processor.get_train_examples(data_dir=self.data_dir)
            train_features, _ = convert_examples_to_features(
                train_examples,
                self.processor.get_labels(),
                self.max_seq_length,
                self.tokenizer,
            )
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
        return train_features

    def _gen_tensor_dataset(self, features):
        all_input_ids = torch.tensor(
            [f.input_ids for f in features],
            dtype=torch.long,
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in features],
            dtype=torch.long,
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features],
            dtype=torch.long,
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in features],
            dtype=torch.long,
        )
        return TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
        )

    def _build_dataset(self):
        features = self._get_train_features()
        return self._gen_tensor_dataset(features)
