import gzip
import json
import os

from tqdm import tqdm

from allennlp.data.dataset_readers.semantic_parsing.wikitables import util


DPD_PATH = "/wikitables/dpd_output/"


def process_file(file_path: str, out_path: str):
    examples = []
    with open(file_path, "r") as data_file:
        for line in tqdm(data_file.readlines()[:200]):
            line = line.strip("\n")
            if not line:
                continue
            parsed_info = util.parse_example_line(line)
            question = parsed_info["question"]
            dpd_output_filename = os.path.join(DPD_PATH, parsed_info["id"] + '.gz')
            try:
                dpd_file = gzip.open(dpd_output_filename)
                sempre_forms = [dpd_line.strip().decode('utf-8') for dpd_line in dpd_file]
            except FileNotFoundError:
                continue
            examples.append((question, sempre_forms))
    with open(out_path, "w") as out_file:
        json.dump(examples, out_file, indent=2)


if __name__ == '__main__':
    process_file("/wikitables/data/random-split-1-train.examples", "train_small.json")
    process_file("/wikitables/data/random-split-1-dev.examples", "dev_small.json")
