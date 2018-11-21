#! /usr/bin/env python
# pylint: disable=invalid-name,bare-except

import gzip
import argparse
import json
import os

from tqdm import tqdm
from weak_supervision.data.dataset_readers.semantic_parsing.wikitables import util


def process_file(file_path: str, out_path: str, lf_path: str, is_labeled=False):
    examples = []
    gold_examples = []
    with open(file_path, "r") as data_file:
        for line in tqdm(data_file.readlines()):
            line = line.strip("\n")
            if not line:
                continue
            if is_labeled:
                try:
                    parsed_info = util.parse_example_line_with_labels(line)
                    sempre_form_gold = parsed_info['target_lf']
                except:
                    continue
            else:
                parsed_info = util.parse_example_line(line)
            question = parsed_info["question"]
            lf_output_filename = os.path.join(lf_path, parsed_info["id"] + '.gz')
            try:
                lf_file = gzip.open(lf_output_filename)
                if is_labeled:
                    sempre_forms = [sempre_form_gold] + [lf_line.strip().decode('utf-8') for lf_line in lf_file]
                else:
                    sempre_forms = [lf_line.strip().decode('utf-8') for lf_line in lf_file]
            except FileNotFoundError:
                continue
            if is_labeled:
                gold_examples.append((question, sempre_form_gold))
            examples.append((question, sempre_forms))
    with open(out_path, "w") as out_file:
        json.dump(examples, out_file, indent=2)

    if is_labeled:
        with open(out_path + "gold", "w") as out_file:
            json.dump(gold_examples, out_file, indent=2)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("train_src", type=str, help="src for creating training data")
    argparser.add_argument("val_src", type=str, help="src for creating validation data")
    argparser.add_argument("dest_dir", type=str, help="dest for dumping processed data")
    argparser.add_argument("lf_dir", type=str, help="Path to original set of logical forms")
    argparser.add_argument('--val_labeled', action="store_true", help="is the src for validation data labeled?")

    args = argparser.parse_args()
    # dump all commandline args
    f = open(f"{args.dest_dir}/preprocess_command.txt", "w")
    f.write(str(args))
    f.close()
    process_file(args.train_src, f"{args.dest_dir}/train.json", args.lf_dir)
    process_file(args.val_src, f"{args.dest_dir}/validation.json", args.lf_dir, args.val_labeled)
