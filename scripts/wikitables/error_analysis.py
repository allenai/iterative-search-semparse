#! /usr/bin/env python


# pylint: disable=invalid-name,wrong-import-position,protected-access
import sys
import os
import gzip
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))


from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util 
from weak_supervision.state_machines import BeamSearch

from weak_supervision.data.dataset_readers import WikiTablesVariableFreeDatasetReader
from weak_supervision.state_machines import GreedyEpsilonBeamSearch

def analyse(input_examples_file: str,
            tables_directory: str,
            archived_model_file_1: str,
            mml: bool = False
            ) -> None:

    reader = WikiTablesVariableFreeDatasetReader(tables_directory=tables_directory,
                                                     keep_if_no_logical_forms = True,
                                                     output_agendas = not mml)
    dataset = reader.read(input_examples_file)
    archive_1 = load_archive(archived_model_file_1)
    model1 = archive_1.model
    model1.eval()

    lines = open(input_examples_file).readlines()

    cnt = 0.0
    if mml:
        f = open('mml.txt', 'w')
    else:
        f = open('erm.txt', 'w')


    for example_line, instance in zip(lines, dataset):
        outputs = model1.forward_on_instance(instance)

    
        parsed_info = util.parse_example_line(example_line)
        example_id = parsed_info["id"]
        question = parsed_info["question"]
        c_1 = len(outputs['correct_logical_form']) > 0

        f.write('%s : %s\n' %(question, str(c_1))) 
            
    f.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", type=str, help="Input file")
    argparser.add_argument("--tables_directory", type=str, help="Tables directory", default = "/u/murtyjay/WikiTableQuestions")
    argparser.add_argument("archived_model_1", type=str, help="Archived model1")
    argparser.add_argument("--use_mml",action='store_true')
    args = argparser.parse_args()
    analyse(args.input, args.tables_directory, args.archived_model_1, args.use_mml) 
