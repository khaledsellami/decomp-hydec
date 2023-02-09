from typing import List, Callable, Set

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_next(myset: Set):
    if len(myset)==0:
        return 0
    for i in range(0, int(max(myset))):
        if i not in myset:
            return i
    return max(myset) + 1


class AbstractAnalysis:
    def __init__(self, features, atoms: List):
        self.features = features
        self.atoms = atoms
        self.supported_atoms = [] # to be defined by each implementation
        self.support_map = [self.atoms.index(i) for i in self.supported_atoms]

    def aggregate(self, current_clusters: List[int]):
        raise NotImplementedError

    def calculate_similarity(self, features):
        raise NotImplementedError

    def map(self, analysis_clusters: List[int], current_clusters: List[int]):
        assert len(current_clusters) >= len(analysis_clusters)
        new_clusters = [
            self.support_map.index(i) if i in self.support_map else c for i, c in enumerate(current_clusters)
        ]
        return new_clusters

    def generate_clusters(self, clustering_function: Callable = None, seed_clusters: List[int] = None):
        if seed_clusters is not None:
            current_clusters = [seed_clusters[i] for i in self.support_map]
            features, aggregation_map = self.aggregate(current_clusters)
        else:
            current_clusters = seed_clusters
            features = self.features
            aggregation_map = np.identity(len(self.support_map))
        sim = self.calculate_similarity(features)
        dist = 1 - sim
        dist[dist < 0] = 0
        db = clustering_function(dist)
        clusters = db.labels_.copy()
        index_set = set(clusters)
        if -1 in index_set:
            index_set.remove(-1)
        for i in range(len(clusters)):
            if clusters[i]==-1:
                n = get_next(index_set)
                index_set.add(n)
                clusters[i] = n
        clusters = clusters.dot(aggregation_map.T)
        cluster_map = {i: c for i, c in zip(current_clusters, clusters)}
        new_clusters = list()
        index_set = set(clusters)
        for c in seed_clusters:
            if c in cluster_map:
                new_clusters.append(cluster_map[c])
            else:
                n = get_next(index_set)
                index_set.add(n)
                new_clusters.append(n)
        return new_clusters


