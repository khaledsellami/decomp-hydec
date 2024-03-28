from typing import List

import numpy as np
from sklearn.cluster import DBSCAN

from .analysis.abstractAnalysis import AbstractAnalysis


class HybridDecomp:
    """
    The class that handles the hybrid and hierarchical clustering process.

    Attributes
    ----------
    analysis_pipeline : the pipeline for the classes responsible for the analysis phases. More than 1 element results
                        in the hybrid process.
    atoms: The list classes/methods to decompose.
    epsilons: the epsilon/max_epsilon values needed for the clusterings (must be same size as analysis_pipeline)
    max_iterations: maximum number of iterations. Only used with the "alternating" strategy.
    min_samples: minimum number of samples required for DBSCAN and for tagging outliers.
    epsilon_step: the step to increment the epsilon after each layer.
    strategy: the combination strategy.
    include_outliers: a flag whether to include outliers in results or not.
    """
    DEFAULT_EPSILON = 0.5
    ALLOWED_STRATEGIES = ["alternating", "alternating_epsilon", "sequential"]

    def __init__(self, analysis_pipeline: List[AbstractAnalysis],
                 atoms: List,
                 epsilons: List[float] = None,
                 max_iterations: int = 50,
                 min_samples: int = 2,
                 epsilon_step: float = 0.05,
                 strategy="alternating",
                 include_outliers: bool = False):
        if epsilons is not None and len(analysis_pipeline) != len(epsilons):
            raise ValueError("analysis_pipeline and epsilons must have the same shape. Found {} and {}".format(
                len(analysis_pipeline), len(epsilons)
            ))
        if strategy not in self.ALLOWED_STRATEGIES:
            raise ValueError("Unrecognized strategy {}. Possible strategies are {}".format(strategy,
                                                                                           self.ALLOWED_STRATEGIES))
        self.analysis_pipeline = analysis_pipeline
        self.atoms = atoms
        if epsilons is None:
            epsilons = [self.DEFAULT_EPSILON for i in range(len(analysis_pipeline))]
        self.epsilons = epsilons
        self.max_iterations = max_iterations
        self.min_samples = min_samples
        self.strategy = strategy
        self.epsilon_step = epsilon_step
        self.include_outliers = include_outliers

    def alternating_clustering(self) -> List[List[int]]:
        """
        Generate the decomposition layers using the alternating strategy: alternating between each analysis approach (
        in a round-robin) until the layers do not change anymore or the maximum number of iterations has been reached.
        :return: the decomposition layers
        """
        layers = list()
        clusters = [i for i in range(len(self.atoms))]
        layers.append(clusters)
        prev_clusters = None
        for i in range(self.max_iterations):
            for analysis, epsilon in zip(self.analysis_pipeline, self.epsilons):
                clustering_function = lambda dist: DBSCAN(
                    eps=epsilon, min_samples=self.min_samples, metric="precomputed").fit(dist).labels_.copy()
                clusters = analysis.generate_clusters(clustering_function, clusters)
                layers.append(list(clusters))
            if prev_clusters is not None and clusters == prev_clusters:
                break
            else:
                prev_clusters = clusters
        return layers

    def alternating_epsilon_clustering(self) -> List[List[int]]:
        """
        Generate the decomposition layers using the alternating epsilon strategy: alternating between each analysis
        approach (in a round-robin) while incrementing their respective epsilons until their limits were reached.
        :return: the decomposition layers
        """
        layers = list()
        clusters = [i for i in range(len(self.atoms))]
        layers.append(clusters)
        # prev_clusters = None
        epsilons = [1e-10 for e in self.epsilons]
        stop_cond = True
        while stop_cond:
            stop_cond = False
            for i in range(len(self.analysis_pipeline)):
                analysis = self.analysis_pipeline[i]
                epsilon = epsilons[i]
                clustering_function = lambda dist: DBSCAN(
                    eps=epsilon, min_samples=self.min_samples, metric="precomputed").fit(dist).labels_.copy()
                clusters = analysis.generate_clusters(clustering_function, clusters)
                layers.append(list(clusters))
                new_epsilon = epsilon + self.epsilon_step
                if new_epsilon >= self.epsilons[i]:
                    epsilons[i] = self.epsilons[i]
                else:
                    epsilons[i] = new_epsilon
                    stop_cond = True
        return layers

    def sequential_clustering(self) -> List[List[int]]:
        """
        Generate the decomposition layers using the sequential strategy: fully clustering each analysis approach before
        moving to the next.
        :return: the decomposition layers
        """
        layers = list()
        clusters = [i for i in range(len(self.atoms))]
        layers.append(clusters)
        for analysis, max_epsilon in zip(self.analysis_pipeline, self.epsilons):
            epsilon = 1e-10
            while epsilon < max_epsilon:
                clustering_function = lambda dist: DBSCAN(
                    eps=epsilon, min_samples=self.min_samples, metric="precomputed").fit(dist).labels_.copy()
                clusters = analysis.generate_clusters(clustering_function, clusters)
                layers.append(list(clusters))
                epsilon = min(epsilon + self.epsilon_step, max_epsilon)
        return layers

    def cluster(self) -> List[List[int]]:
        """
        Generate the decomposition layers based on the provided strategy.
        :return: the decomposition layers
        """
        if self.strategy == "alternating":
            layers = self.alternating_clustering()
        elif self.strategy == "alternating_epsilon":
            layers = self.alternating_epsilon_clustering()
        elif self.strategy == "sequential":
            layers = self.sequential_clustering()
        else:
            raise NotImplementedError
        if self.include_outliers:
            layers = self.tag_outliers(layers)
        return layers

    def tag_outliers(self, layers: List[List[int]]) -> List[List[int]]:
        """
        Tag outliers atoms with -1. An outlier is any atom that belongs to a cluster with a size lower than min_samples.

        :param layers: decomposition layers
        :return: decomposition layer with outlier atoms tagged with the value -1
        """
        new_layers = list()
        for layer in layers:
            unique, counts = np.unique(layer, return_counts=True)
            layer = np.array(layer)
            layer[np.isin(layer, unique[counts < self.min_samples])] = -1
            new_layers.append(list(layer))
        return new_layers


