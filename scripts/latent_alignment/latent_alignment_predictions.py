"""
This script is just for random toying with model - running predictions and so on. No guarantees
of maintenance.
"""
#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position,protected-access
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.models.archival import load_archive
from weak_supervision.data.dataset_readers import LatentAlignmentDatasetReader


def make_data(input_examples_file: str,
              archived_model_file: str) -> None:
    reader = LatentAlignmentDatasetReader(max_logical_forms=200)
    dataset = reader.read(input_examples_file)
    archive = load_archive(archived_model_file)
    model = archive.model
    model.eval()


    with open(input_examples_file, "r") as data_file:
        examples = json.load(data_file)

    baseline_acc = 0.0
    acc = 0.0
    total = 0.0
    for example, instance in zip(examples, dataset):
        utterance, sempre_forms = example
        if len(sempre_forms[0]) <= 10:
            baseline_acc += 1.0
        print(utterance)
        outputs = model.forward_on_instance(instance)
        if outputs["most_similar"] == sempre_forms[0]:
            acc += 1.0
        total += 1.0


    print(baseline_acc/total)
    print(acc/total)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", type=str, help="Input file")
    argparser.add_argument("archived_model", type=str, help="Archived model.tar.gz")
    args = argparser.parse_args()
    make_data(args.input, args.archived_model)
