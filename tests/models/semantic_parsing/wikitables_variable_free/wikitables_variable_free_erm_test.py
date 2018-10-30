# pylint: disable=no-self-use
import pytest
from flaky import flaky

from allennlp.common.testing import ModelTestCase

from weak_supervision import *

@pytest.mark.java
class WikiTablesVariableFreeErmTest(ModelTestCase):
    def setUp(self):
        super(WikiTablesVariableFreeErmTest, self).setUp()
        self.set_up_model("fixtures/semantic_parsing/wikitables_variable_free/experiment-erm.json",
                          "fixtures/data/wikitables/sample_data.examples")

    @flaky
    def test_model_can_train_save_and_load(self):
        # We have very few embedded actions on our agenda, and so it's rare that this parameter
        # actually gets used.  We know this parameter works from our NLVR ERM test, so it's easier
        # to just ignore it here than to try to finagle the test to make it so this has a non-zero
        # gradient.
        ignore = {'_decoder_step._checklist_multiplier'}
        self.ensure_model_can_train_save_and_load(self.param_file, gradients_to_ignore=ignore)
