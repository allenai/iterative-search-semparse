#! /usr/bin/env python
# pylint: disable=invalid-name,wrong-import-position,protected-access
import sys
import os
import gzip
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util 
from weak_supervision.semparse.executors import WikiTablesVariableFreeExecutor

from weak_supervision.data.dataset_readers import WikiTablesVariableFreeDatasetReader
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import WordTokenizer
from weak_supervision.semparse.contexts import TableQuestionContext
from weak_supervision.data.dataset_readers.semantic_parsing.wikitables import util
from numpy.random import choice


join_token = {"filter_in": "is",
               "filter_date_equals" : "is",
               "filter_number_equals" : "equals",
               "filter_number_lesser" :  "is less than",
               "filter_number_greater" : "is greater than",
               "filter_date_lesser" : "is before",
               "filter_date_greater" : "is after"}

join_token_how_many = {"filter_in": "equal to",
               "filter_date_equals" : "equal to",
               "filter_number_equals" : "equal to",
               "filter_number_lesser" :  "less than",
               "filter_number_greater" : "greater than",
               "filter_date_lesser" : "before",
               "filter_date_greater" : "after"}




def gen_date(date):
    day_str = ""
    if date.day == 1:
        day_str = "1st"
    elif date.day == 2:
        day_str = "2nd"
    elif date.day ==3:
        day_str = "3rd"
    elif date.day > 1:
        day_str = "{}th".format(date.day)

    str2month = {
        "" : -1,
        'january': 1,
        'february': 2,
        'march': 3,
        'april': 4,
        'may': 5,
        'june': 6,
        'july': 7,
        'august': 8,
        'september': 9,
        'october': 10,
        'november': 11,
        'december': 12,
        }

    month2str = {str2month[month] : month for month in str2month} 
    month_str = month2str[date.month]
    date_str = ""
    if day_str: 
        date_str += day_str
    if month_str:
        date_str += " "
        date_str += month_str
    if date.year != -1:
        date_str += " "
        date_str += str(date.year)
    return date_str

"Define simple templates"
def select_template(x_lf, x_lang, col2, first_or_last=None):
    col2_str = " ".join(col2.split("_"))

    if first_or_last:
        question = "What is the {} {} for which {}".format(first_or_last, col2_str, x_lang)
        lf = '(select_string ({} {}) string_column:{})'.format(first_or_last, x_lf, col2)
    else:
        question = "What is the {} for which {}".format(col2_str, x_lang)
        lf = '(select_string {} string_column:{})'.format(x_lf, col2)
    return lf, question

def how_many_fiter_template(x_lf, x_lang):
    lf = "(count {})".format(x_lf)
    question = "How many times is {}".format(x_lang)
    return lf, question

def arg_max_min_template(compare_col, obj_col, dsl_token):
    lf = "(select_string ({} all_rows number_column:{}) string_column:{})".format(dsl_token, compare_col, obj_col)
    if dsl_token == "argmax": dsl_str = "largest"
    if dsl_token == "argmin": dsl_str = "smallest"
    obj_col_str = " ".join(obj_col.split("_"))
    compare_col_str = " ".join(compare_col.split("_"))
    question = "Which {} has the {} {}".format(obj_col_str, dsl_str, compare_col_str)
    return lf, question


def filter_template(formal_col, col, obj, obj_str, dsl_token, join_dict = join_token):
    col = " ".join(col.split("_"))
    lf = "({} all_rows {} {})".format(dsl_token, formal_col, obj) 
    question = "{} {} {}".format(col, 
                                 join_dict[dsl_token],
                                 obj_str)
    return lf, question

def generate_question_and_lf(table_context):
    chosen_row = random.choice(table_context.table_data) 
    all_cols = chosen_row.keys()
    all_col_names = []
    type2cols = {'number' : [], 'date' : [], 'string' : []}

    for col in all_cols:
        col_type, col = col.split(":")
        if col == 'null' or 'notes' in col:
            continue
        if col_type == 'string_column':
            type2cols['string'].append(col)
        elif col_type == 'number_column':
            type2cols['number'].append(col)
        elif col_type == 'date_column':
            type2cols['date'].append(col)
        all_col_names.append(col)

    # remove redundancies
    to_remove = []
    for col in type2cols['string']:
        if col in type2cols['date']:
            to_remove.append(col)
        elif col in type2cols['number']:
            to_remove.append(col)

    type2cols['string'] = list(filter(lambda col : col not in to_remove, type2cols['string']))
    to_remove = []
    for col in type2cols['number']:
        if col in type2cols['date']:
            to_remove.append(col)

    type2cols['number'] = list(filter(lambda col : col not in to_remove, type2cols['number']))
    
    all_ops = list(join_token.keys())
    if len(type2cols['date']) == 0:
        all_ops = list(filter(lambda obj: 'date' not in obj, all_ops))
    if len(type2cols['string']) == 0:
        all_ops = list(filter(lambda obj: 'string' not in obj, all_ops))
    if len(type2cols['number']) == 0:
        all_ops = list(filter(lambda obj: 'number' not in obj, all_ops))

    # argmax/argmin/proceed to select/count

    argmax_argmin = choice(['argmax','argmin','select_count'], 1, [0.05, 0.05, 0.9])[0]
    if argmax_argmin in ['argmax', 'argmin']:
        if len(type2cols['number']) == 0: return None, None
        chosen_col = random.choice(type2cols['number'])
        if len(type2cols['string']) == 0: return None, None
        chose_col2 = random.choice(type2cols['string'])
        return arg_max_min_template(chosen_col, chose_col2 ,argmax_argmin)

    choose_op = random.choice(all_ops) 
    if 'date' in choose_op:
        col_type = 'date'
    elif 'number' in choose_op:
        col_type = 'number'
    else:
        col_type = 'string'
   
    if len(type2cols[col_type]) == 0:
        return None, None
    chosen_col = random.choice(type2cols[col_type])
    formal_name = '{}_column:{}'.format(col_type,chosen_col)

    chosen_cell = chosen_row[formal_name]
    if chosen_cell == None:
        return None, None

    chose_col2 = random.choice(all_col_names)
    if chose_col2 == chosen_col:
        return None, None

    if col_type == 'date':
        cell_str = gen_date(chosen_cell)
        chosen_cell = "(date {} {} {})".format(chosen_cell.year, chosen_cell.month, chosen_cell.day)
    elif col_type == 'string':
        cell_str = chosen_cell
        chosen_cell = "string:{}".format(cell_str)
        cell_str = " ".join(cell_str.split("_")) 
    else:
        cell_str = chosen_cell
        
    first_or_last = choice(['first','last',None], 1, [0.15,0.15,0.7])[0]
    select_or_count = choice(['select','count'], 1, [0.8, 0.2])[0] 
    if select_or_count == 'select':
        lf1, q1 =  filter_template(formal_name, chosen_col, chosen_cell, cell_str, choose_op)
        return select_template(lf1, q1, chose_col2, first_or_last)
    else:
        lf1, q1 =  filter_template(formal_name, chosen_col, chosen_cell, cell_str, choose_op, join_token_how_many)
        return how_many_fiter_template(lf1, q1)

def create_synthetic_data(table_context):
    q_lf = set()
    trials = 0
    while len(q_lf) != args.num_lf:
        question, lf = generate_question_and_lf(table_context)
        trials += 1
        if trials >= 1000: break
        if question is not None:
            q_lf.add((question, lf))
    return q_lf
    return []


def create_data(input_examples_file: str,
            tables_directory: str) -> None:

    reader = WikiTablesVariableFreeDatasetReader(tables_directory=tables_directory,
                                                 keep_if_no_logical_forms = True)
    dataset = reader.read(input_examples_file)
    lines = open(input_examples_file).readlines()

    cnt = 0.0
    f = open('synthetic_data.examples', 'w')
    tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))

    _id = 1000
    for example_line, instance in zip(lines, dataset):
        parsed_info = util.parse_example_line(example_line)
        table_name = parsed_info["table_filename"]
        table_filename = os.path.join(tables_directory,
                                      table_name.replace("csv", "tagged"))
        table_lines = [line.split("\t") for line in open(table_filename).readlines()]
        question = parsed_info['question']
        tokenized_question = tokenizer.tokenize(question.lower())
        table_context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
        executor = WikiTablesVariableFreeExecutor(table_context.table_data)
        data = create_synthetic_data(table_context)
        for (lf, question) in data:
            answer = executor.execute(lf)
            if isinstance(answer, list):
                if len(answer) == 0 or all(tok == '' for tok in answer): continue
                else:
                    target_val = " ".join(['(description "{}")'.format(val) for val in answer])
            else:
                target_val = '(description "{}")'.format(answer)
            line = '(example (id nt-{}) (utterance "{}?") (context (graph tables.TableKnowledgeGraph {})) (targetValue (list {})) (targetFormula {}))'.format(_id, question, table_name, target_val, lf) 
            parsed_info = util.parse_example_line_with_labels(line, use_lang=True)
            assert parsed_info['target_lf'] == lf
            assert parsed_info['table_filename'] == table_name
            print(parsed_info['question'] , question)
            f.write("%s\n" %line)
            _id += 1
            
    f.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", type=str, help="Input file")
    argparser.add_argument("--tables_directory", type=str, help="Tables directory", default = "/u/murtyjay/WikiTableQuestions")
    argparser.add_argument("--num_lf", type=int, help="LF per table", default = 100)

    args = argparser.parse_args()
    create_data(args.input, args.tables_directory) 
