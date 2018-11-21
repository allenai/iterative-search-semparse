from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def preprocess_tokens(token):
    if token.startswith('fb:row.row.'):
        return token.replace('fb:row.row.', '')
    elif token.startswith('fb:cell.'):
        return token.replace('fb:cell.', '')
    elif token.startswith('fb:part.'):
        return token.replace('fb:part.', '')
    else:
        return token


@DatasetReader.register("latent_alignment")
class LatentAlignmentDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 utterance_token_indexers: Dict[str, TokenIndexer] = None,
                 logical_form_token_indexers: Dict[str, TokenIndexer] = None,

                 max_logical_forms: int = 500,
                 process_tokens: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_logical_forms = max_logical_forms
        self._utterance_token_indexers = utterance_token_indexers or \
                                         {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._logical_form_token_indexers = logical_form_token_indexers or \
                                            {'lf_tokens': SingleIdTokenIndexer(namespace='lf_tokens')}

        self._process_tokens = process_tokens
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as data_file:
            examples = json.load(data_file)
        for utterance, sempre_forms in examples:
            if len(sempre_forms) > self._max_logical_forms:
                sempre_form_gold = sempre_forms[0]
                sempre_forms_dpd = sempre_forms[1:]
                sempre_forms_dpd.sort(key=len)
                sempre_forms = [sempre_form_gold] + sempre_forms_dpd[:self._max_logical_forms]
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
            if self._process_tokens:
                logical_form_tokens = [preprocess_tokens(token) for token in
                                       logical_form.replace('(', '').replace(')', '').split(' ')]
            else:
                logical_form_tokens = logical_form.replace('(', '').replace(')', '').split(' ')
            logical_form_fields.append(TextField([Token(t) for t in logical_form_tokens],
                                                 self._logical_form_token_indexers))
        fields["logical_forms"] = ListField(logical_form_fields)
        fields["logical_form_strings"] = MetadataField(logical_forms)
        fields["utterance_string"] = MetadataField(utterance)
        return Instance(fields)
