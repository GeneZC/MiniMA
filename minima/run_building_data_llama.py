# -*- coding: utf-8 -*-

"""Build LM TFRecord examples for LLaMA."""

import os
import re
import math
import glob
import json
import argparse
import random

from transformers import LlamaTokenizer

from multiprocessing import Pool

from data import TFRecordWriter


BUFSIZE = 40960000 * 2


class Example:
    """An example (sentence)."""
    def __init__(self, tokens):
        self.tokens = tokens
        # self.segments = segments

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([str(x) for x in self.tokens]))
        # s += "segments: %s\n" % (" ".join([str(x) for x in self.segments]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def build_examples_from_documents(documents, max_seq_length, rng):
    # Account for [BOS] or [EOS].
    # max_seq_length = max_seq_length - 1

    examples = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(documents):
        segment = documents[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(documents) - 1 or current_length >= max_seq_length:
            if current_chunk: # This is kind of redundant.
                tokens = []
                accum_length = 0
                for j in range(len(current_chunk)):
                    tokens.extend(current_chunk[j])
                    accum_length += len(current_chunk[j])
                    if accum_length >= max_seq_length:
                        break
                tokens = tokens[:max_seq_length]
                example = Example(
                    tokens=tokens,
                )
                examples.append(example)
            i -= len(current_chunk) - j - 1
            # Reuse unused partial segments if possible.
            if accum_length > max_seq_length:
                current_chunk = [current_chunk[j][max_seq_length - accum_length:]]
                current_length = len(current_chunk[0])
            else:
                current_chunk = []
                current_length = 0
        i += 1
    # There could be something left from the last round, so here we go.
    while current_chunk:
        tokens = []
        tokens.extend(current_chunk[-1])
        accum_length = len(current_chunk[-1])
        tokens = tokens[:max_seq_length]
        example = Example(
            tokens=tokens,
        )
        examples.append(example)
        # Reuse unused partial segments if possible.
        if accum_length > max_seq_length:
            current_chunk = [current_chunk[-1][max_seq_length - accum_length:]]
        else:
            current_chunk = []
    return examples


def worker(lines, tokenizer, max_seq_length, rng):
    documents = []
    # Input file format:
    # One document per line.
    for line in lines:
        line = line.strip()
        try:
            line = json.loads(line)
            if isinstance(line, list):
                line = line[0]
            elif isinstance(line, dict):
                line = line["text"]
        except: # str.
            pass
        try:
            tokens = tokenizer.tokenize(line) + [tokenizer.eos_token] + [tokenizer.bos_token]
            documents.append(tokens)
        except:
            pass

    # Remove empty documents.
    documents = [x for x in documents if x]
    # Shuffle documents.
    rng.shuffle(documents)

    examples = build_examples_from_documents(
        documents,
        max_seq_length,
        rng,
    )

    # Shuffle examples.
    rng.shuffle(examples)
    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory for input files to format."
    )
    parser.add_argument(
        "--input_regex", 
        type=str, 
        required=True, 
        help="Regex for input files to format."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory for output files to write."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        required=True, 
        help="The tokenizer to use.",
    )
    parser.add_argument(
        "--do_lower_case", 
        action="store_true", 
        help="Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models."
    )
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Maximum sequence length.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed.")
    parser.add_argument("--num_processors", type=int, default=8,
                        help="Num of processors.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=args.do_lower_case)
    rng = random.Random(args.seed)

    pool = Pool(args.num_processors)
    for input_file in glob.glob(os.path.join(args.input_dir, args.input_regex)):
        print("*** Building examples ***")
        print(f"   from {input_file}")

        stream = open(input_file, "r")
        output_file = os.path.join(args.output_dir, os.path.basename(input_file) + ".tfrecord")
        print("*** Writing examples ***")
        print(f"   to {output_file}")
        num_examples = 0
        with TFRecordWriter(output_file) as writer:
            while True:
                lines = stream.readlines(BUFSIZE)
                if not lines:
                    break
                chunk_size = len(lines) // args.num_processors + 1
                arguments = [(lines[i * chunk_size: (i + 1) * chunk_size], tokenizer, args.max_seq_length, rng) 
                            for i in range(args.num_processors)]
                gathered_examples = pool.starmap(worker, arguments)
                if not gathered_examples:
                    continue
                all_examples = []
                for examples in gathered_examples:
                    all_examples.extend(examples)
                for example in all_examples:
                    writer.write({
                        "indices": (tokenizer.convert_tokens_to_ids(example.tokens), "int"),
                        # "segments": (example.segments, "int"),
                    })
                    # description = {"indices": "int", "segments": "int"}
                    num_examples += 1
                    if num_examples <= 5:
                        print(example)
                print(f"  Having written {num_examples} examples", )
        stream.close()		

if __name__ == "__main__":
    main()
