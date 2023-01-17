from typing import List

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .abstractAnalysis import AbstractAnalysis


class TfidfAnalysis(AbstractAnalysis):
    SUPPORTED_AGGREGATION = ["mean", "sum"]

    def __init__(self, tfidf_vectors: np.ndarray, atoms: List, supported_atoms: List,
                 atom_tokens: List[List[str]] = None, aggregation="mean"):
        if aggregation not in self.SUPPORTED_AGGREGATION:
            raise ValueError("Unsupported aggregation method \"{}\"".format(aggregation))
        self.aggregation = aggregation
        self.features = tfidf_vectors
        self.atom_tokens = atom_tokens
        self.atoms = atoms
        self.supported_atoms = supported_atoms
        self.support_map = [self.atoms.index(i) for i in self.supported_atoms]

    def aggregate(self, current_clusters: List[int]):
        aggregation_map = OneHotEncoder(sparse=False, dtype=int).fit_transform(
            np.array(current_clusters).reshape(-1, 1)
        )
        if self.aggregation == "mean":
            features = aggregation_map.T.dot(self.features)/aggregation_map.sum(axis=0).reshape(-1,1)
        elif self.aggregation == "sum":
            features = aggregation_map.T.dot(self.features)
        return features, aggregation_map

    def calculate_similarity(self, features: np.ndarray):
        return features.dot(features.T)
