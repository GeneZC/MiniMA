# -*- coding: utf-8 -*-

from data.tfrecord_io import (
    TFRecordReader, 
    TFRecordWriter,
)
from data.pipelines import (
    LlamaLMDataPipeline,
    LlamaInstructDataPipeline,
)


PIPELINE_CLASS = {
    "llama_lm": LlamaLMDataPipeline,
    "llama_instruct": LlamaInstructDataPipeline,
}

def get_pipeline_class(data_type):
    return PIPELINE_CLASS[data_type]


import math
import warnings
import numpy as np

import torch.distributed as dist
from torch.utils.data import IterableDataset


class TFRecordDistributedDataset(IterableDataset):
    def __init__(self, data, queue_size=4096, num_replicas=None, rank=None, shuffle=True):
        super().__init__()
        self.data = data
        self.queue_size = queue_size
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        # Do ceiling to make the data evenly divisible among devices.
        self.num_instances = math.ceil(len(self.data) / self.num_replicas)
        self.total_num_instances = self.num_instances * self.num_replicas
    
    def __len__(self):
        return self.num_instances

    def __iter__(self):
        stream = self.data.stream(shard_info=(self.rank, self.num_replicas))
        buffer = []
        try:
            for _ in range(self.queue_size):
                buffer.append(next(stream))
        except StopIteration:
            warnings.warn("Number of elements in the iterator is less than the "
                        f"queue size (N={self.queue_size}).")
        while buffer:
            if self.shuffle:
                index = np.random.randint(len(buffer))
            else:
                index = 0
            try:
                item = buffer[index]
                buffer[index] = next(stream)
                yield item
            except StopIteration:
                yield buffer.pop(index)
        
    def __del__(self):
        self.data.close()


class TFRecordDataset(IterableDataset):
    def __init__(self, data, queue_size=4096, shuffle=True):
        super().__init__()
        self.data = data
        self.queue_size = queue_size
        self.shuffle = shuffle
        self.num_instances = len(self.data)
    
    def __len__(self):
        return self.num_instances

    def __iter__(self):
        stream = self.data.stream()
        buffer = []
        try:
            for _ in range(self.queue_size):
                buffer.append(next(stream))
        except StopIteration:
            warnings.warn("Number of elements in the iterator is less than the "
                        f"queue size (N={self.queue_size}).")
        while buffer:
            if self.shuffle:
                index = np.random.randint(len(buffer))
            else:
                index = 0
            try:
                item = buffer[index]
                buffer[index] = next(stream)
                yield item
            except StopIteration:
                yield buffer.pop(index)

    def __del__(self):
        self.data.close()