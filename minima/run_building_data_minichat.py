# -*- coding: utf-8 -*-

"""Build LM TFRecord examples for MiniChat."""

import os
import json
import math
import glob
import argparse
import random

from transformers import LlamaTokenizer

from multiprocessing import Pool

from data import TFRecordWriter


BUFSIZE = 40960000 * 2


class Example:
    """An example (sentence)."""
    def __init__(self, tokens, mask):
        self.tokens = tokens
        self.mask = mask

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([str(x) for x in self.tokens]))
        s += "mask: %s\n" % (" ".join([str(x) for x in self.mask]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def accum_len(lst):
        return sum([len(e) for e in lst])


def truncate_conversation(text_a_tokens, text_b_tokens, max_length):
    """Truncate a conversational input in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    a_idx, b_idx = 0, 0
    while True:
        total_length = accum_len(text_a_tokens) + accum_len(text_b_tokens)
        if total_length <= max_length:
            break
        if accum_len(text_a_tokens) > accum_len(text_b_tokens):
            while len(text_a_tokens[a_idx]) == 1: # Avoid [EOS].
                a_idx += 1
            text_a_tokens[a_idx].pop(-2) # Avoid [EOS].
        else:
            while len(text_b_tokens[b_idx]) == 1:
                b_idx += 1
            text_b_tokens[b_idx].pop(-2)


def worker(lines, tokenizer, max_seq_length, rng):
    meta = "‘MiniChat’是一个由‘Beccurio’开发的AI语言模型。下面是人类和MiniChat之间的一段对话。MiniChat的回复应当尽可能详细，并且以Markdown的形式输出。MiniChat应当拒绝参与违背伦理的讨论。"
    meta += "</s>"
    meta_tokens = tokenizer.tokenize(meta)
    examples = []
    num_skipped_examples = 0
    # Input file format:
    # (1) One conversation per line. These should ideally contain one or more turns.
    # (2) Each turn has "User", "Assistant".
    for line in lines:
        line = json.loads(line.strip())
        user_prefix_turns_tokens = []
        user_suffix_turns_tokens = []
        assistant_prefix_turns_tokens = []
        assistant_suffix_turns_tokens = []
        for turn in line:
            user_prefix_turns_tokens.append(tokenizer.tokenize("[|User|]"))
            user_suffix_turns_tokens.append(tokenizer.tokenize(turn["User"]) + [tokenizer.eos_token])
            assistant_prefix_turns_tokens.append(tokenizer.tokenize("[|Assistant|]"))
            assistant_suffix_turns_tokens.append(tokenizer.tokenize(turn["Assistant"]) + [tokenizer.eos_token])

        # Frustratingly, for too long a sequence of tokens (e.g., >> 2048), we have to drop tokens of the assistant.
        try:
            preserved_length = len(meta_tokens) + accum_len(user_prefix_turns_tokens)+ accum_len(assistant_prefix_turns_tokens)
            truncate_conversation(user_suffix_turns_tokens, assistant_suffix_turns_tokens, max_seq_length - preserved_length)
        except:
            num_skipped_examples += 1
            continue
        
        tokens = [] + meta_tokens # Kind of `copy()`.
        mask = len(meta_tokens) * [0]
        for i in range(len(line)):
            tokens += user_prefix_turns_tokens[i] + user_suffix_turns_tokens[i] + assistant_prefix_turns_tokens[i] + assistant_suffix_turns_tokens[i]
            mask += len(user_prefix_turns_tokens[i]) * [0] + len(user_suffix_turns_tokens[i]) * [0] + len(assistant_prefix_turns_tokens[i]) * [0] + len(assistant_suffix_turns_tokens[i]) * [1]
        
        example = Example(
            tokens=tokens,
            mask=mask,
        )
        examples.append(example)
    return (examples, num_skipped_examples,)


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

        DEVSIZE = int(os.path.getsize(input_file) * 0.001)
        stream = open(input_file, "r")
        output_file = os.path.join(args.output_dir, os.path.basename(input_file) + f".dev" + ".tfrecord")
        print("*** Writing examples ***")
        print(f"   to {output_file}")
        num_examples = 0
        with TFRecordWriter(output_file) as writer:
            lines = stream.readlines(DEVSIZE)
            # if not lines:
            #     break
            all_examples, all_num_skipped_examples = worker(lines, tokenizer, args.max_seq_length, rng)
            for example in all_examples:
                writer.write({
                    "indices": (tokenizer.convert_tokens_to_ids(example.tokens), "int"),
                    "mask": (example.mask, "int"),
                })
                # description = {"indices": "int", "mask": "int"}
                num_examples += 1
                if num_examples <= 5:
                    print(example)
            print(f"  Having written {num_examples} examples")
            print(f"  Having skipped {all_num_skipped_examples} examples")
        output_file = os.path.join(args.output_dir, os.path.basename(input_file) + f".train" + ".tfrecord")
        print("*** Writing examples ***")
        print(f"   to {output_file}")
        num_examples = 0
        with TFRecordWriter(output_file) as writer:
            while True:
                lines = stream.readlines(BUFSIZE)
                if not lines:
                    break
                chunk_size = math.ceil(len(lines) / args.num_processors)
                arguments = [(lines[i * chunk_size: (i + 1) * chunk_size], tokenizer, args.max_seq_length, rng) 
                            for i in range(args.num_processors)]
                gathered_examples = pool.starmap(worker, arguments)
                all_examples = []
                all_num_skipped_examples = 0
                for examples in gathered_examples:
                    all_examples.extend(examples[0])
                    all_num_skipped_examples += examples[1]
                for example in all_examples:
                    writer.write({
                        "indices": (tokenizer.convert_tokens_to_ids(example.tokens), "int"),
                        "mask": (example.mask, "int"),
                    })
                    # description = {"indices": "int", "mask": "int"}
                    num_examples += 1
                    if num_examples <= 5:
                        print(example)
                print(f"  Having written {num_examples} examples")
                print(f"  Having skipped {all_num_skipped_examples} examples")
        stream.close()		

if __name__ == "__main__":
    main()
