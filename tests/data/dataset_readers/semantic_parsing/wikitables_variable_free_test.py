# pylint: disable=no-self-use

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase

from weak_supervision.data.dataset_readers import WikiTablesVariableFreeDatasetReader
from weak_supervision.semparse.worlds import WikiTablesVariableFreeWorld


def assert_dataset_correct(dataset):
    instances = list(dataset)
    assert len(instances) == 2
    instance = instances[0]

    assert instance.fields.keys() == {
            'question',
            'table',
            'world',
            'actions',
            'target_action_sequences',
            'target_values',
            }

    question_tokens = ["what", "was", "the", "last", "year", "where", "this", "team", "was", "a",
                       "part", "of", "the", "usl", "a", "-", "league", "?"]
    assert [t.text for t in instance.fields["question"].tokens] == question_tokens


    # The content of this will be tested indirectly by checking the actions; we'll just make
    # sure we get a WikiTablesWorld object in here.
    assert isinstance(instance.fields['world'].as_tensor({}), WikiTablesVariableFreeWorld)

    action_fields = instance.fields['actions'].field_list
    actions = [action_field.rule for action_field in action_fields]

    # We should have been able to read all of the logical forms in the file.  If one of them can't
    # be parsed, or the action sequences can't be mapped correctly, the DatasetReader will skip the
    # logical form, log an error, and keep going (i.e., it won't crash).
    num_action_sequences = len(instance.fields["target_action_sequences"].field_list)
    assert num_action_sequences == 10

    # We should have sorted the logical forms by length.  This is the action sequence
    # corresponding to the shortest logical form in the examples _by tree size_, which is _not_ the
    # first one in the file, or the shortest logical form by _string length_.  It's also a totally
    # made up logical form, just to demonstrate that we're sorting things correctly.
    action_sequence = instance.fields["target_action_sequences"].field_list[0]
    action_indices = [l.sequence_index for l in action_sequence.field_list]
    actions = [actions[i] for i in action_indices]
    assert actions == ['@start@ -> n',
                       'n -> [<r,<f,n>>, r, f]',
                       '<r,<f,n>> -> average',
                       'r -> [<r,r>, r]',
                       '<r,r> -> last',
                       'r -> [<r,<t,<s,r>>>, r, t, s]',
                       '<r,<t,<s,r>>> -> filter_in',
                       'r -> all_rows',
                       't -> string_column:league',
                       's -> string:usl_a_league',
                       'f -> number_column:year']


class WikiTablesVariableFreeDatasetReaderTest(AllenNlpTestCase):
    def test_reader_reads(self):
        offline_search_directory = "fixtures/data/wikitables/action_space_walker_output"
        params = {
                'lazy': False,
                'tables_directory': "fixtures/data/wikitables",
                'offline_logical_forms_directory': offline_search_directory,
                }
        reader = WikiTablesVariableFreeDatasetReader.from_params(Params(params))
        dataset = reader.read("fixtures/data/wikitables/sample_data.examples")
        assert_dataset_correct(dataset)

    def test_reader_reads_with_lfs_in_tarball(self):
        offline_search_directory = \
                "fixtures/data/wikitables/action_space_walker_output_with_single_tarball"
        params = {
                'lazy': False,
                'tables_directory': "fixtures/data/wikitables",
                'offline_logical_forms_directory': offline_search_directory,
                }
        reader = WikiTablesVariableFreeDatasetReader.from_params(Params(params))
        dataset = reader.read("fixtures/data/wikitables/sample_data.examples")
        assert_dataset_correct(dataset)
