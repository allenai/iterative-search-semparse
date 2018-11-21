#! /usr/bin/env python
# pylint: disable=invalid-name,bare-except,unused-variable

from typing import Dict

GLOBAL_MAPPING = {"@index": ["fb:row.row.index"],
                  "@!index": ["(", "reverse", "fb:row.row.index", ")"],
                  "@!p.num": ["(", "reverse", "fb:cell.cell.number", ")"],
                  "@p.num": ["fb:cell.cell.number"],
                  "@!p.num2": ["(", "reverse", "fb:cell.cell.num2", ")"],
                  "@p.num2": ["fb:cell.cell.num2"],
                  "@!p.date": ["(", "reverse", "fb:cell.cell.date", ")"],
                  "@p.date": ["fb:cell.cell.date"],
                  "@type": ["fb:type.object.type"],
                  "@row": ["fb:type.row"],
                  "@p.part": ["fb:cell.cell.part"],
                  "@!p.part": ["(", "reverse", "fb:cell.cell.part", ")"],
                  "@next": ["fb:row.row.next"],
                  "@!next": ["(", "reverse", "fb:row.row.next", ")"],
                 }

def translate_to_lambda_dcs(formula: str):
    formula = formula.replace("(", "( ").replace(")", " )")
    parts = formula.split()
    if parts[1] == "targetFormula":
        # We don't need the outermost nesting in this lisp string
        parts = parts[2:-1]
    translated_parts = []
    for part in parts:
        part_is_num = False
        try:
            part_num = float(part)
            part_is_num = True
        except:
            pass
        if part_is_num:
            # This needs translation
            translated_parts += ["(", "number", part, ")"]
        elif "." in part or "@" in part:
            # This needs translation too.
            if part in GLOBAL_MAPPING:
                translated_parts += GLOBAL_MAPPING[part]
            else:
                if "r." in part:
                    entity_name = part.split("r.")[-1]
                    canonical_name = f"fb:row.row.{entity_name}"
                elif "c." in part:
                    entity_name = part.split("c.")[-1]
                    canonical_name = f"fb:cell.{entity_name}"
                elif "q." in part:
                    entity_name = part.split("q.")[-1]
                    canonical_name = f"fb:part.{entity_name}"
                else:
                    raise RuntimeError(f"Cannot handle entity: {part}")
                if "!" in part:
                    translated_parts += ["(", "reverse", canonical_name, ")"]
                else:
                    translated_parts += [canonical_name]
        else:
            translated_parts += [part]
    translated_parts = " ".join(translated_parts)
    translated_parts = translated_parts.replace("( ", "(").replace(" )", ")")
    return translated_parts



def parse_example_line_with_labels(lisp_string: str) -> Dict:
    """
    Training data in WikitableQuestions comes with examples in the form of lisp strings in the format:
        (example (id <example-id>)
                 (utterance <question>)
                 (context (graph tables.TableKnowledgeGraph <table-filename>))
                 (targetValue (list (description <answer1>) (description <answer2>) ...)))
                 (targetFormula (formula))

    We parse such strings and return the parsed information here.
    """
    id_piece, rest = lisp_string.split(') (utterance "')
    example_id = id_piece.split('(id ')[1]
    question, rest = rest.split('") (context (graph tables.TableKnowledgeGraph ')
    table_filename, rest = rest.split(')) (targetValue (list')
    target_value_strings, rest = rest.split("(targetFormula")
    target_value_strings = target_value_strings.strip().split("(description")
    target_values = []
    for string in target_value_strings:
        string = string.replace(")", "").replace('"', '').strip()
        if string != "":
            target_values.append(string)

    return {'id': example_id,
            'question': question,
            'table_filename': table_filename,
            'target_values': target_values,
            'target_lf' : translate_to_lambda_dcs(rest.strip()[:-2])}

def parse_example_line(lisp_string: str) -> Dict:
    """
    Training data in WikitableQuestions comes with examples in the form of lisp strings in the format:
        (example (id <example-id>)
                 (utterance <question>)
                 (context (graph tables.TableKnowledgeGraph <table-filename>))
                 (targetValue (list (description <answer1>) (description <answer2>) ...)))

    We parse such strings and return the parsed information here.
    """
    id_piece, rest = lisp_string.split(') (utterance "')
    example_id = id_piece.split('(id ')[1]
    question, rest = rest.split('") (context (graph tables.TableKnowledgeGraph ')
    table_filename, rest = rest.split(')) (targetValue (list')
    target_value_strings = rest.strip().split("(description")
    target_values = []
    for string in target_value_strings:
        string = string.replace(")", "").replace('"', '').strip()
        if string != "":
            target_values.append(string)
    return {'id': example_id,
            'question': question,
            'table_filename': table_filename,
            'target_values': target_values}
