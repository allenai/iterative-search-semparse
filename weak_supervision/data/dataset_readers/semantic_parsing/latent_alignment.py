from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("latent_alignment")
class LatentAlignmentDatasetReader(DatasetReader):
    def __init__(self,
                 max_logical_forms: int = 500,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_logical_forms = max_logical_forms
        self._utterance_token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._logical_form_token_indexers = {'lf_tokens': SingleIdTokenIndexer(namespace='lf_tokens')}
        self._tokenizer = WordTokenizer(JustSpacesWordSplitter())

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as data_file:
            examples = json.load(data_file)
        for utterance, sempre_forms in examples:
            if len(sempre_forms) > self._max_logical_forms:
                sempre_forms.sort(key=len)
                sempre_forms = sempre_forms[:self._max_logical_forms]
            yield self.text_to_instance(utterance, sempre_forms)

    @overrides
    def text_to_instance(self,  # type: ignore
                         utterance: str,
                         logical_forms: List[str]) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokens = TextField(self._tokenizer.tokenize(utterance), self._utterance_token_indexers)
        fields["utterance"] = tokens

        logical_form_fields = []
        for logical_form in logical_forms:
            logical_form_tokens = logical_form.replace('(', '').replace(')', '').split(' ')
            logical_form_fields.append(TextField([Token(t) for t in logical_form_tokens],
                                                 self._logical_form_token_indexers))
        fields["logical_forms"] = ListField(logical_form_fields)
        fields["logical_form_strings"] = MetadataField(logical_forms)
        fields["utterance_string"] = MetadataField(utterance)

        return Instance(fields)
