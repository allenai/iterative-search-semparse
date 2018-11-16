# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common.testing import ModelTestCase

from weak_supervision import *

class LatentIbmModel1Test(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model("fixtures/latent_ibm_model_1/experiment.json",
                          "fixtures/data/wikitables/alignment_preprocessed.json")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
