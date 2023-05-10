from typing import List

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .abstractAnalysis import AbstractAnalysis
from .similarity import call_similarity, cousage_similarity


class DependencyAnalysis(AbstractAnalysis):
    SIMILARITY_MAP = {"call": call_similarity, "cousage": cousage_similarity}

    def __init__(self, features: np.ndarray, atoms: List, supported_atoms: List, similarity="call"):
        if similarity not in self.SIMILARITY_MAP:
            raise ValueError("Unsupported similarity function \"{}\"".format(similarity))
        self.similarity = similarity
        #!!!TOFIX
        if len([i for i in supported_atoms if not i in atoms])>0:
            self.supported_atoms = [i for i in supported_atoms if i in atoms]
            supported_atoms_map = [i for i, j in enumerate(supported_atoms) if j in atoms]
            self.features = features[supported_atoms_map][:, supported_atoms_map]
        else:
            self.features = features
            self.supported_atoms = supported_atoms
        self.atoms = atoms
        self.support_map = [self.atoms.index(i) for i in self.supported_atoms]

    def aggregate(self, current_clusters: List[int]):
        aggregation_map = OneHotEncoder(sparse=False, dtype=int).fit_transform(
            np.array(current_clusters).reshape(-1, 1)
        )
        C = aggregation_map.T
        return C.dot(self.features).dot(C.T), aggregation_map

    def calculate_similarity(self, features):
        return self.SIMILARITY_MAP[self.similarity](np.array(features))