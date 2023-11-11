# -*- coding: utf-8 -*-

import io
import os
import glob
import struct
import math

try:
    import crc32c
except ImportError:
    crc32c = None

import numpy as np
import data.example_pb2 as protobuf

import subprocess


class TFRecordWriter:
    """Opens a TFRecord file for writing.

    Params:
    -------
    record_path: str
        Path to the tfrecord file.
    """

    def __init__(self, record_path):
        self.record_writer = io.open(record_path, "wb")
        self.record_index_writer = io.open(record_path + ".index", "w")
        self.written_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the tfrecord file."""
        self.record_writer.close()
        self.record_index_writer.close()

    def write(self, datum):
        """
        Write an example into tfrecord file.

        Params:
        -------
        datum: dict
            Dictionary of tuples of form (value, dtype). dtype can be
            "float" or "int".
        """
        record = TFRecordWriter.serialize_tf_example(datum)

        length = len(record)
        length_bytes = struct.pack("<Q", length)
        self.record_writer.write(length_bytes)
        self.record_writer.write(TFRecordWriter.masked_crc(length_bytes))
        self.record_writer.write(record)
        self.record_writer.write(TFRecordWriter.masked_crc(record))
        record_bytes = 8 + 4 + length + 4
        self.record_index_writer.write(str(self.written_bytes) + " " + str(record_bytes) + "\n")
        self.written_bytes += record_bytes

    @staticmethod
    def masked_crc(data):
        """CRC checksum."""
        mask = 0xa282ead8
        crc = crc32c.crc32(data)
        masked = ((crc >> 15) | (crc << 17)) + mask
        masked = np.uint32(masked & np.iinfo(np.uint32).max)
        masked_bytes = struct.pack("<I", masked)
        return masked_bytes

    @staticmethod
    def serialize_tf_example(datum):
        """Serialize example into tfrecord.Example proto.

        Params:
        -------
        datum: dict
            Dictionary of tuples of form (value, dtype). dtype can be
            "float" or "int".

        Returns:
        --------
        proto: bytes
            Serialized tfrecord.example to bytes.
        """
        feature_map = {
            "byte": lambda f: protobuf.Feature(
                bytes_list=protobuf.BytesList(value=f)),
            "float": lambda f: protobuf.Feature(
                float_list=protobuf.FloatList(value=f)),
            "int": lambda f: protobuf.Feature(
                int64_list=protobuf.Int64List(value=f))
        }

        def serialize(value, dtype):
            if not isinstance(value, (list, tuple, np.ndarray)):
                value = [value]
            return feature_map[dtype](value)

        features = {key: serialize(value, dtype) for key, (value, dtype) in datum.items()}
        example_proto = protobuf.Example(features=protobuf.Features(feature=features))
        return example_proto.SerializeToString()


def index(path, proportion):
    ret = []
    with open(path, "r") as f:
        for line in f:
            ret.append(int(line.strip().split()[0]))
    if proportion is not None:
        for k, p in proportion.items():
            if k in path:
                p_int = math.floor(p)
                p_dec = p - p_int
                ret = ret * p_int + ret[:math.ceil(p_dec * len(ret))]
                break
    return ret


def resplit(record_reader_group, record_index_group, max_num_records_per_split=1024):
    new_record_reader_group, new_record_index_group = [], []
    for record_reader, record_index in zip(record_reader_group, record_index_group):
        num_splits = math.ceil(len(record_index) / max_num_records_per_split)
        for i in range(num_splits):
            new_record_reader_group.append(record_reader)
            new_record_index_group.append(record_index[i * max_num_records_per_split: (i + 1) * max_num_records_per_split])
    zipped_group = list(zip(new_record_reader_group, new_record_index_group))
    np.random.shuffle(zipped_group)
    return zip(*zipped_group)


class TFRecordReader:
    """Opens a TFRecord file for reading.

    Params:
    -------
    record_path: str
        Path (with necessary regex) to the tfrecord file(s).
    """
    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)
    typename_map = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }
    def __init__(self, record_path_or_regex, proportion=None, description=None):
        self.record_path_group = glob.glob(record_path_or_regex)
        self.record_reader_group = [io.open(record_path, "rb") for record_path in self.record_path_group]
        self.record_index_group = [index(record_path + ".index", proportion) for record_path in self.record_path_group]
        # Split each index into smaller ones, and achieve a better shuffle by the way.
        self.record_reader_group, self.record_index_group = resplit(self.record_reader_group, self.record_index_group)
        self.num_records_group = [len(record_index) for record_index in self.record_index_group]
        self.group_size = len(self.num_records_group)
        self.total_num_records = sum(self.num_records_group)
        self.description = description

    def __len__(self):
        return self.total_num_records

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the tfrecord file."""
        for record_reader in self.record_reader_group:
            record_reader.close()

    def stream(self, shard_info=None):
        if shard_info is None:
            for i in range(self.group_size):
                record_reader = self.record_reader_group[i]
                record_index = self.record_index_group[i]
                yield from self._read_records(record_reader, record_index[0], record_index[-1], self.description)
        else:
            shard_idx, num_shards = shard_info
            shard_num_records = math.ceil(self.total_num_records / num_shards)
            # The index behaves like [beg_index, end_index).
            beg_index = shard_num_records * shard_idx
            end_index = shard_num_records * (shard_idx + 1)
            for i in range(self.group_size):
                num_records = self.num_records_group[i]
                record_reader = self.record_reader_group[i]
                record_index = self.record_index_group[i]
                # Skip this stream.
                if beg_index >= num_records:
                    beg_index -= num_records
                    end_index -= num_records
                    continue
                # Read from this stream.
                if end_index <= num_records: # Records are enough in this stream.
                    yield from self._read_records(record_reader, record_index[beg_index], record_index[end_index - 1], self.description)
                    break
                else: # Records are not enough in this stream, continue reading from anther stream.
                    yield from self._read_records(record_reader, record_index[beg_index], record_index[-1], self.description)
                    beg_index = 0
                    end_index -= num_records
            # If we break the loop, then finish; otherwise, add extra records to make it evenly divisible.
            else:
                # Reuse the last record reader, the loop should only repeat a few times.
                while True:
                    if beg_index >= num_records:
                        beg_index -= num_records
                        end_index -= num_records
                        continue
                    if end_index <= num_records:
                        yield from self._read_records(record_reader, record_index[beg_index], record_index[end_index - 1], self.description)
                        break
                    else:
                        yield from self._read_records(record_reader, record_index[beg_index], record_index[-1], self.description)
                        beg_index = 0
                        end_index -= num_records   

    @staticmethod
    def _read_records(record_reader, beg_byte, end_byte, description):
        record_reader.seek(beg_byte)
        while record_reader.tell() <= end_byte:
            if record_reader.readinto(TFRecordReader.length_bytes) != 8:
                raise RuntimeError("Failed to read the record size.")
            if record_reader.readinto(TFRecordReader.crc_bytes) != 4:
                raise RuntimeError("Failed to read the beg token.")
            length, = struct.unpack("<Q", TFRecordReader.length_bytes)
            if length > len(TFRecordReader.datum_bytes):
                TFRecordReader.datum_bytes = TFRecordReader.datum_bytes.zfill(int(length * 1.5))
            datum_bytes_view = memoryview(TFRecordReader.datum_bytes)[:length]
            if record_reader.readinto(datum_bytes_view) != length:
                raise RuntimeError("Failed to read the record.")
            if record_reader.readinto(TFRecordReader.crc_bytes) != 4:
                raise RuntimeError("Failed to read the end token.")
            
            example = protobuf.Example()
            example.ParseFromString(datum_bytes_view)
            yield TFRecordReader.extract_feature(example, description) 

    @staticmethod
    def extract_feature(example, description):
        features = example.features.feature

        all_keys = list(features.keys())

        if description is None or len(description) == 0:
            description = dict.fromkeys(all_keys, None)
        elif isinstance(description, list):
            description = dict.fromkeys(description, None)

        processed_features = {}
        for key, typename in description.items():
            if key not in all_keys:
                raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")

            processed_features[key] = TFRecordReader.process_feature(features[key], key, typename)
        return processed_features

    @staticmethod
    def process_feature(feature, key, typename):
        # NOTE: We assume that each key in the example has only one field
        # (either "bytes_list", "float_list", or "int64_list")!
        field = feature.ListFields()[0]
        inferred_typename, value = field[0].name, field[1].value

        if typename is not None:
            tf_typename = TFRecordReader.typename_map[typename]
            if tf_typename != inferred_typename:
                reversed_map = {v: k for k, v in TFRecordReader.typename_map.items()}
                raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                            f"(should be '{reversed_map[inferred_typename]}').")

        if inferred_typename == "bytes_list":
            value = value[0]
        elif inferred_typename == "float_list":
            value = np.array(value, dtype=np.float32)
        elif inferred_typename == "int64_list":
            value = np.array(value, dtype=np.int64)
        return value