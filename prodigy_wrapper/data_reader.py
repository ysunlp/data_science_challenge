# coding: utf8
# from __future__ import unicode_literals, print_function

import random
import json
import numpy as np
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from overrides import overrides
from typing import Dict
from typing import List

class IMDBDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.
    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}
    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``
    where the ``label`` is derived from the venue of the paper.
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 label: List[str] = None,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.labels = label

    @overrides
    def _read(self, file_path):
#         content = JSONL(file_path)
#         for line in content:
        with open(cached_path(file_path), "r") as data_file:
#             logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                review = json.loads(line)
                text = review['text']
                if(len(self.labels) == 1): # binary classification with one label
                    label = 1 if review['answer']=='accept' else 0
                elif(len(self.labels) == 2): # binary classification with two labels
                    label == 1 if (review['label']==self.labels[0] and review['answer'] == 'accept') or (review['label']==self.labels[1] and review['answer'] == 'reject') else 0
                else: #multi-classification, need modification
                    label = np.zeros(self.labels)
                    for j in range(len(self.labels)):
                        if (review['label']==self.labels[j] and review['answer'] == 'accept'):
                            label[j] == 1 
                        else:
                            # didn't consider reject situation, should assign -1 if the label is reject to offer information to model.
                            label[j] == 0
                yield self.text_to_instance(text,label)

    @overrides
    def text_to_instance(self, text: str, label: int) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field, 'length':LabelField(text_field.sequence_length(),skip_indexing=True),'label': LabelField(label,skip_indexing=True)}
        
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'SemanticScholarDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers)