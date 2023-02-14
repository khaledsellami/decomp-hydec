from typing import List, Callable, Set, Union

import numpy as np
from sklearn.preprocessing import OneHotEncoder


class AbstractAnalysis:
    """ abstract class that represents the analysis phase in the decomposition process """
    def __init__(self, features, atoms: List, supported_atoms: List = []):
        self.features = features
        self.atoms = atoms
        self.supported_atoms = supported_atoms # to be defined by each implementation
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

    def generate_clusters(self, clustering_function: Callable = None, seed_clusters: List[int] = None) -> List[int]:
        """
        Generates a new clustering based on existing clusters by aggregating the available features and calculating the
        similarity.

        :param clustering_function: the clustering algorithm function to call. It needs to take as input a distance
        matrix and return the clusters vector. -1 refers to outliers
        :param seed_clusters: the clusters that will seed into the next clustering step with shape (len(self.atoms)).
        :return: the new clusters with shape (len(self.atoms))
        """
        if seed_clusters is not None:
            # if seed clusters were provided
            # include only the clusters of the supported atoms
            assert len(seed_clusters) == len(self.atoms)
            supported_clusters = [seed_clusters[i] for i in self.support_map]
            # get the aggregated features and the aggregation map (a NxM one-hot matrix where N is the number of atoms
            # and M is the number of clusters)
            features, aggregation_map = self.aggregate(supported_clusters)
            assert aggregation_map.shape[0] == len(supported_clusters)
        else:
            # if clustering from scratch
            supported_clusters = seed_clusters
            features = self.features
        assert supported_clusters is None or len(supported_clusters) == len(self.support_map)
        # calculate similarity and distance
        sim = self.calculate_similarity(features)
        dist = 1 - sim
        dist[dist < 0] = 0
        # generate new clusters
        clusters = clustering_function(dist)
        if supported_clusters is None:
            new_clusters = [
                clusters[self.support_map.index(i)] if i in self.support_map else -1 for i in range(len(self.atoms))
            ]
        else:
            # start by replacing outlier indexes (-1) with unique indexes
            clusters[clusters == -1] = list(
                set(np.arange(clusters.shape[0])) - set(np.unique(clusters)))[:(clusters == -1).sum()]
            # clusters[clusters==-1] = np.arange(np.max(clusters)+1, np.max(clusters)+1+(clusters==-1).sum())
            # project new clusters on the initial atoms
            clusters = clusters.dot(aggregation_map.T)
            new_clusters = self.project_to_original(clusters, seed_clusters)
        assert len(new_clusters) == len(self.atoms)
        return new_clusters

    def project_to_original(self, clusters: np.ndarray, seed_clusters: List[int]) -> List[int]:
        """
        Projects the new clusters to the original atoms.

        :param clusters: new clusters (size = len(self.supported_atoms))
        :param seed_clusters: The clusters that were fed to the clustering step (size = len(self.atoms))
        :return: new and old clusters including all atoms  (size = len(self.atoms))
        """
        assert len(clusters) == len(self.support_map)
        # map the new clustering to the original atoms
        # start by mapping the old cluster indexes to the new ones
        cluster_map = {seed_clusters[i]: c for i, c in zip(self.support_map, clusters)}
        # initialize an empty vector (-2 refers to unassigned cluster)
        new_clusters = np.array([-2 for i in seed_clusters])
        # assign the new clusters to the supported atoms
        new_clusters[self.support_map] = clusters
        # The following line extracts the atoms that are not supported but share a cluster with a supported atom.
        joined_clusters_map = {
            i: cluster_map[j] for i, j in enumerate(seed_clusters) if i not in self.support_map and j in cluster_map
        }
        # assign the new clusters indexes to the unsupported atoms that belong to them.
        new_clusters[list(joined_clusters_map.keys())] = list(joined_clusters_map.values())
        # assign new unique indexes to the old clusters that are not mapped to new clusters
        old_indexes = set(np.unique(seed_clusters)) - set(cluster_map)
        new_indexes = list(set(np.arange(len(seed_clusters))) - set(np.unique(clusters)))[:len(old_indexes)]
        indexes_map = dict(zip(old_indexes, new_indexes))
        new_clusters[new_clusters == -2] = [indexes_map[v] for v in np.array(seed_clusters)[new_clusters == -2]]
        return list(new_clusters)


