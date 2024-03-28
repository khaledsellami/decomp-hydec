from typing import List

import numpy as np

from .abstractAnalysis import AbstractAnalysis


class SumAnalysis(AbstractAnalysis):
    """
    assumes all analysis classes support the same atoms
    """
    def __init__(self, analysis_classes: List[AbstractAnalysis], weights: List[float] = None):
        if weights is not None:
            assert np.sum(weights) == 1
            if len(analysis_classes) != len(weights):
                raise ValueError("analysis_classes and weights must have the same shape. Found {} and {}".format(
                    len(analysis_classes), len(weights)
                ))
        assert len(analysis_classes) > 1
        assert all([
            len(analysis.supported_atoms) == len(analysis_classes[0].supported_atoms) for analysis in analysis_classes
        ])
        self.weights = weights
        self.analysis_classes = analysis_classes
        self.atoms = analysis_classes[0].atoms
        self.supported_atoms = analysis_classes[0].supported_atoms
        self.support_map = analysis_classes[0].support_map
        self.features = [analysis.features for analysis in analysis_classes]

    def aggregate(self, current_clusters: List[int]):
        new_features = list()
        for analysis in self.analysis_classes:
            new_feature, aggregation_map = analysis.aggregate(current_clusters)
            new_features.append(new_feature)
        return new_features, aggregation_map

    def calculate_similarity(self, features):
        sims = np.array([analysis.calculate_similarity(f) for analysis, f in zip(self.analysis_classes, features)])
        if self.weights is not None:
            sims = sims*np.array(self.weights).reshape(-1, 1, 1)
            sim = np.sum(sims, axis=0)
        else:
            sim = np.mean(sims, axis=0)
        sim[sim > 1] = 1
        return sim



