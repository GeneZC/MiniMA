# -*- coding: utf-8 -*-

import os
import time
import glob
import shutil
import logging
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_kwargs_to_config(config, **kwargs):
    for k in kwargs:
        setattr(config, k, kwargs[k])


def singleton(cls):
    _instance = {}
    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner


@singleton
class Logger(logging.Logger):
    def __init__(self):
        super().__init__("miniformers")

    def add_stream_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s"))
        self.addHandler(sh)

    def add_file_handler(self, save_dir):
        fh = logging.FileHandler(save_dir + "/log.txt", "w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s"))
        self.addHandler(fh)

    def set_verbosity_info(self):
        self.setLevel(logging.INFO)

    def set_verbosity_error(self):
        self.setLevel(logging.ERROR)


class AverageMeter:
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.buffer = []

    def reset(self):
        self.buffer = []

    def update(self, val):
        self.buffer.append(val)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    @property
    def avg(self):
        return sum(self.buffer) / len(self.buffer)


def keep_recent_ckpt(output_dir, recent_cnt=3):
    dir_tuples = [(d, time.strptime("-".join(d.split("-")[-4:]), "%Y-%m-%d-%H:%M:%S")) for d in glob.glob(os.path.join(output_dir, r"ckpt-*"))]
    dir_tuples = sorted(dir_tuples, key=lambda x: x[1])[:-recent_cnt]
    for d, _ in dir_tuples:
        shutil.rmtree(d)


def find_most_recent_ckpt(output_dir):
    dir_tuples = [(d, time.strptime("-".join(d.split("-")[-4:]), "%Y-%m-%d-%H:%M:%S")) for d in glob.glob(os.path.join(output_dir, r"ckpt-*"))]
    if len(dir_tuples) == 0:
        return None
    else:
        return sorted(dir_tuples, key=lambda x: x[1])[-1][0]
    