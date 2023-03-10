import itertools
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, DataCollatorWithPadding, BertModel
from datasets import Dataset

from .abstractAnalysis import AbstractAnalysis
from .similarity import cosine_similarity
from .embeddingModel import EmbeddingModel


class WordEmbeddingAnalysis(AbstractAnalysis):
    SUPPORTED_AGGREGATION = ["mean", "sum"]

    def __init__(self, atom_tokens: List[List[str]], model: EmbeddingModel, atoms: List, supported_atoms: List,
                 aggregation="mean", embeddings=None):
        if aggregation not in self.SUPPORTED_AGGREGATION:
            raise ValueError("Unsupported aggregation method \"{}\"".format(aggregation))
        # change type of word to those without preprocessing
        super().__init__(atom_tokens, atoms, supported_atoms)
        self.model = model
        self.aggregation = aggregation
        self.embeddings = None
        if embeddings is not None:
            self.embeddings = embeddings
        elif aggregation != "combine":
            self.embeddings = self.get_embeddings(self.features)

    def get_embeddings(self, features):
        embeddings = np.zeros((len(features), self.model.dim))
        for i, tokens in enumerate(features):
            embeddings[i, :] = self.model.get_embedding_vector(tokens)
        return embeddings

    def aggregate(self, current_clusters: List[int]):
        aggregation_map = OneHotEncoder(sparse=False, dtype=int).fit_transform(
            np.array(current_clusters).reshape(-1, 1)
        )
        if self.aggregation == "mean":
            features = aggregation_map.T.dot(self.embeddings)/aggregation_map.sum(axis=0).reshape(-1,1)
        elif self.aggregation == "sum":
            features = aggregation_map.T.dot(self.embeddings)
        elif self.aggregation == "combine":
            new_words = list()
            for j in range(aggregation_map.shape[1]):
                print(np.where(aggregation_map[:, j])[0])
                new_words.append(list(itertools.chain(*[self.features[i] for i in np.where(aggregation_map[:, j])[0]])))
            features = self.get_embeddings(new_words)
        else:
            raise ValueError("Unsupported aggregation method \"{}\"".format(self.aggregation))
        return features, aggregation_map

    def calculate_similarity(self, features: np.ndarray):
        return cosine_similarity(features)
