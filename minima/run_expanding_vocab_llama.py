
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
import re
import argparse

from transformers import LlamaTokenizer

import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm


# 。，？！；：“”‘’《》〈〉【】『』—…「」
chinese_punctuation = r'[\u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u300c\u300d]'


def is_all_chinese(_str):
    for _char in _str:
        if not ('\u4e00' <= _char <= '\u9fa5' or re.search(chinese_punctuation, _char) is not None):
            return False
    return True


def is_contain_chinese(_str):
    for _char in _str:
        if '\u4e00' <= _char <= '\u9fa5' or re.search(chinese_punctuation, _char) is not None:
            return True
    return False


parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default="llama", type=str)
parser.add_argument('--chinese_sp_model_file', default="./chinese_sp.model", type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
chinese_sp_model_file = args.chinese_sp_model_file

# Load sentencepiece protobufs.
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(chinese_sp_model_file)

llama_proto = sp_pb2_model.ModelProto()
llama_proto.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
chinese_proto = sp_pb2_model.ModelProto()
chinese_proto.ParseFromString(chinese_sp_model.serialized_model_proto())

# Print number of tokens.
print(len(llama_tokenizer), len(chinese_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

# Add Chinese tokens to LLaMA tokenizer.
llama_proto_tokens_set = set(p.piece for p in llama_proto.pieces)
min_score = llama_proto.pieces[-1].score
trashed_pieces = []
print(len(llama_proto_tokens_set))
print(f"Before: {len(llama_proto_tokens_set)}")
for p in chinese_proto.pieces:
    piece = p.piece
    if piece not in llama_proto_tokens_set:
        if is_contain_chinese(piece):
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            min_score -= 1
            new_p.score = min_score
            llama_proto.pieces.append(new_p)
        elif not any(x.isdigit() for x in piece):
            # new_p = sp_pb2_model.ModelProto().SentencePiece()
            # new_p.piece = piece
            # min_score -= 1
            # new_p.score = min_score
            trashed_pieces.append(piece)
# Add pieces from trash bin to make the vocab divisible by 64 (for most parallelism concerns).
num_to_add = (len(llama_proto.pieces) // 64 + 1) * 64 - len(llama_proto.pieces)
print(trashed_pieces[:num_to_add])
# ['......', ')、', '?<', ':<', 'QQ', 'iPhone', 'NBA', 'HK', '!<', ')<', '◆', 'formula', 'CEO', 'GDP', 'MV', 'VR', '-{', 'SUV', 'Vi', 'CPU', 'mg']
for piece in trashed_pieces[:num_to_add]:
    new_p = sp_pb2_model.ModelProto().SentencePiece()
    new_p.piece = piece
    min_score -= 1
    new_p.score = min_score
    llama_proto.pieces.append(new_p)
print(f"After: {len(llama_proto.pieces)}")
# After: 49216

# Save.
output_dir = "llama-ada"
os.makedirs(output_dir, exist_ok=True)
with open(output_dir + '/tokenizer.model', 'wb') as f:
    f.write(llama_proto.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_dir + "/tokenizer.model")

tokenizer.save_pretrained(output_dir)
print(f"LLaMA-ada tokenizer has been saved to {output_dir}")

# Test.
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
llama_ada_tokenizer = LlamaTokenizer.from_pretrained(output_dir)

text="白日依山尽，黄河入海流。欲穷千里目，更上一层楼。Let\'s democratize large language models! 101+102=203. Bravo, Assistant!"
print("Test text:\n", text)
print(f"Tokenized by LLaMA tokenizer: {llama_tokenizer.tokenize(text)}")
print(f"Tokenized by LLaMA-ada tokenizer: {llama_ada_tokenizer.tokenize(text)}")
"""
Test text:
 白日依山尽，黄河入海流。欲穷千里目，更上一层楼。Let's democratize large language models! 101+102=203. Bravo, Assistant!
Tokenized by LLaMA tokenizer: ['▁', '白', '日', '<0xE4>', '<0xBE>', '<0x9D>', '山', '<0xE5>', '<0xB0>', '<0xBD>', '，', '黄', '河', '入', '海', '流', '。', '<0xE6>', '<0xAC>', '<0xB2>', '<0xE7>', '<0xA9>', '<0xB7>', '千', '里', '目', '，', '更', '上', '一', '<0xE5>', '<0xB1>', '<0x82>', '<0xE6>', '<0xA5>', '<0xBC>', '。', 'Let', "'", 's', '▁dem', 'ocrat', 'ize', '▁large', '▁language', '▁models', '!', '▁', '1', '0', '1', '+', '1', '0', '2', '=', '2', '0', '3', '.', '▁Bra', 'vo', ',', '▁Ass', 'istant', '!']
Tokenized by LLaMA-ada tokenizer: ['▁白', '日', '依', '山', '尽', '，', '黄河', '入', '海', '流', '。', '欲', '穷', '千里', '目', '，', '更', '上', '一层', '楼', '。', 'Let', "'", 's', '▁dem', 'ocrat', 'ize', '▁large', '▁language', '▁models', '!', '▁', '1', '0', '1', '+', '1', '0', '2', '=', '2', '0', '3', '.', '▁Bra', 'vo', ',', '▁Ass', 'istant', '!']
"""