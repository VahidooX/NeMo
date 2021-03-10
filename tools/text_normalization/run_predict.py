# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
from time import perf_counter
from typing import List

from tools.text_normalization.normalize import normalizers

import json

'''
Runs normalization on text data
'''


def load_file(file_path: str) -> List[str]:
    """
    Load given text file into list of string.
    Args:
        file_path: file path
        input_format: the format of input file: "text_file" or "asr_json_manifest"
    Returns: flat list of string
    """
    res = []
    with open(file_path, 'r') as fp:
        for line in fp:
            res.append(line)
    return res


def write_file(file_path: str, data: List[str]):
    """
    Writes out list of string to file.
    Args:
        file_path: file path
        data: list of string
    """
    with open(file_path, 'w') as fp:
        for line in data:
            fp.write(line + '\n')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", help="input file path", required=True, type=str)
    parser.add_argument("--input_format", choices=['text_file', 'asr_json_manifest'], default='text_file', help="input file path. It can be a text file or a json manifest with a filed named 'text'.", type=str)
    parser.add_argument("--output", help="output file path", required=True, type=str)
    parser.add_argument("--verbose", help="print normalization info. For debugging", action='store_true')
    parser.add_argument(
        "--normalizer", default='nemo', help="normalizer to use (" + ", ".join(normalizers.keys()) + ")", type=str
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.input
    normalizer = normalizers[args.normalizer]

    print("Loading data: " + file_path)
    data = load_file(file_path)
    if args.input_format == "asr_json_manifest":
        text_data = []
        for line_id, line in enumerate(data):
            sample = json.loads(line)
            data[line_id] = sample
            text_data.append(sample["text"].strip())
    else:
        text_data = data
    print("- Data: " + str(len(data)) + " sentences")
    t_start = perf_counter()
    normalizer_prediction = normalizer(text_data, verbose=args.verbose)
    t_end = perf_counter()
    print(f"- Finished in {t_end-t_start} seconds. Processed {len(data)/(t_end-t_start)} sentences per second.")
    print("- Normalized. Writing out...")
    if args.input_format == "asr_json_manifest":
        for line_id, line in enumerate(data):
            print(f'Orig: {data[line_id]["text"]} \nNorm: {normalizer_prediction[line_id]}')
            data[line_id]["text"] = normalizer_prediction[line_id]
            normalizer_prediction[line_id] = json.dumps(data[line_id])

    write_file(args.output, normalizer_prediction)
