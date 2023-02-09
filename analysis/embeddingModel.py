import itertools
from typing import List, Union

import numpy as np
import spacy
import fasttext.util
import gensim.downloader as gensim_api


class EmbeddingModel:
    """ Abstraction layer over the embedding model to use"""
    SUPPORTED_EMBEDDING_MODELS = ["fasttext", "word2vec"]
    SUPPORTED_EMBEDDING_MODELS_SIZE = [300]
    SUPPORTED_POOLING_APPROACHES = ["avg", "max"]
    PUNCT = r". … …… , : ; ! ? ¿ ؟ ¡ ( ) [ ] { } < > _ # * & 。 ？ ！ ， 、 ； ： ～ · । ، ۔ ؛ ٪ " + \
            '\' " ” “ ` ‘ ´ ’ ‚ , „ » « 「 」 『 』 （ ） 〔 〕 【 】 《 》 〈 〉 ' + \
            "- – — -- --- —— ~"
    TO_REMOVE = [""] + PUNCT.strip().split(" ")

    def __init__(self, type: str = "fasttext", dim: int = 300, pooling_approach: str = "avg"):
        assert type in self.SUPPORTED_EMBEDDING_MODELS
        assert dim in self.SUPPORTED_EMBEDDING_MODELS_SIZE
        assert pooling_approach in self.SUPPORTED_POOLING_APPROACHES
        self.type = type
        self.dim = dim
        self.pooling_approach = pooling_approach
        self.model = None
        self.tokenizer = spacy.load("en_core_web_sm")

    def load_embedding_model(self):
        if self.type == "fasttext":
            fasttext.util.download_model('en', if_exists='ignore')
            self.model = fasttext.load_model('cc.en.300.bin')
            if self.dim == 100:
                fasttext.util.reduce_model(self.model, self.dim)
        elif self.type == "word2vec":
            assert self.dim == 300 # work-around since 100 dim word2vec embedding were not found
            self.model = gensim_api.load("word2vec-google-news-{}".format(self.dim))
        return self.model

    def get_embedding_vector(self, tokens: List[str]):
        if self.model is None:
            raise ValueError("Make sure to load the model first!")
        embedding_matrix = np.zeros((len(tokens), self.dim))
        for i, token in enumerate(tokens):
            if self.type == "word2vec" and token not in self.model:
                continue
            embedding_matrix[i, :] = self.model[token]
        if self.pooling_approach == "avg":
            return np.mean(embedding_matrix, axis=0)
        elif self.pooling_approach == "max":
            return np.max(embedding_matrix, axis=0)

    def tokenize(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            return [x.text.lower() for x in self.tokenizer(text) if not x.text.strip() in self.TO_REMOVE]
        elif isinstance(text, list):
            return list(itertools.chain(*[
                [x.text.lower() for x in self.tokenizer(item) if not x.text.strip() in self.TO_REMOVE] for item in text
            ]))
        else:
            raise TypeError("Wrong type {} for argument text. Expected String or List.".format(type(text)))
