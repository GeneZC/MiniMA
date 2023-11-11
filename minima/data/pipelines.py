# -*- coding: utf-8 -*-

import random
import torch
import collections
import numpy as np


class DataPipeline:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        if max_length is None:
            self.max_length = tokenizer.model_max_length
        else:
            self.max_length = max_length

    @staticmethod
    def _pad(indices, max_length, pad_idx):
        """Pad a sequence to the maximum length."""
        pad_length = max_length - len(indices)
        return list(indices) + [pad_idx] * pad_length

    def collate(self, batch):
        raise NotImplementedError()
    

def shift_right(indices, decoder_start_idx):
    """
    Shift indices one token to the right, and add a start.
    """
    shifted_indices = np.zeros_like(indices)
    shifted_indices[1:] = indices[:-1]
    shifted_indices[0] = decoder_start_idx
    return shifted_indices

class LlamaLMDataPipeline(DataPipeline):
    Example = collections.namedtuple(
        "Example", 
        (
            "indices",
        )
    )
    Batch = collections.namedtuple(
        "Batch", 
        (
            "text_indices",
            "text_mask",
            "lm_labels",
        )
    )

    def __init__(self, tokenizer, max_length=None):
        super().__init__(tokenizer, max_length)

    def collate(self, batch):
        batch_text_indices = []
        batch_text_mask = []
        batch_lm_labels = []
        for example in batch:
            example = LlamaLMDataPipeline.Example(**example)
            text_indices = shift_right(example.indices, self.tokenizer.bos_token_id)
            text_mask = [1] * len(text_indices)
            lm_labels = example.indices
            # NOTE: LLaMA has not pad_token, use eos_token instead.
            batch_text_indices.append(self._pad(text_indices, self.max_length, self.tokenizer.eos_token_id))
            batch_text_mask.append(self._pad(text_mask, self.max_length, 0))
            batch_lm_labels.append(self._pad(lm_labels, self.max_length, self.tokenizer.eos_token_id))
        return LlamaLMDataPipeline.Batch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            lm_labels=torch.tensor(batch_lm_labels, dtype=torch.long),
        )


class LlamaInstructDataPipeline(DataPipeline):
    Example = collections.namedtuple(
        "Example", 
        (
            "indices",
            "mask",
        )
    )
    Batch = collections.namedtuple(
        "Batch", 
        (
            "text_indices",
            "text_mask",
            "labels",
            "label_mask"
        )
    )

    def __init__(self, tokenizer, max_length=None):
        super().__init__(tokenizer, max_length)

    def collate(self, batch):
        batch_text_indices = []
        batch_text_mask = []
        batch_labels = []
        batch_label_mask = []
        for example in batch:
            example = LlamaInstructDataPipeline.Example(**example)
            text_indices = shift_right(example.indices, self.tokenizer.bos_token_id)
            text_mask = [1] * len(text_indices)
            labels = example.indices
            label_mask = example.mask
            # NOTE: LLaMA has not pad_token, use eos_token instead.
            batch_text_indices.append(self._pad(text_indices, self.max_length, self.tokenizer.eos_token_id))
            batch_text_mask.append(self._pad(text_mask, self.max_length, 0))
            batch_labels.append(self._pad(labels, self.max_length, self.tokenizer.eos_token_id))
            batch_label_mask.append(self._pad(label_mask, self.max_length, 0))
        return LlamaInstructDataPipeline.Batch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            labels=torch.tensor(batch_labels, dtype=torch.long),
            label_mask=torch.tensor(batch_label_mask, dtype=torch.bool),
        )
    