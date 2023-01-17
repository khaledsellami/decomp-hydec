from typing import List

from sklearn.cluster import DBSCAN

from analysis.abstractAnalysis import AbstractAnalysis


class HybridDecomp:
    DEFAULT_EPSILON = 0.5

    def __init__(self, analysis_pipeline: List[AbstractAnalysis],
                 atoms: List,
                 epsilons: List[float] = None,
                 max_iterations: int = 50,
                 min_samples: int = 2,
                 strategy="iterative"):
        if epsilons is not None and len(analysis_pipeline) != len(epsilons):
            raise ValueError("analysis_pipeline and epsilons must have the same shape. Found {} and {}".format(
                len(analysis_pipeline), len(epsilons)
            ))
        self.analysis_pipeline = analysis_pipeline
        self.atoms = atoms
        if epsilons is None:
            epsilons = [self.DEFAULT_EPSILON for i in range(len(analysis_pipeline))]
        self.epsilons = epsilons
        self.max_iterations = max_iterations
        self.min_samples = min_samples
        self.strategy = strategy

    def iterative_clustering(self):
        layers = list()
        clusters = [i for i in range(len(self.atoms))]
        layers.append(clusters)
        prev_clusters = None
        for i in range(self.max_iterations):
            for analysis, epsilon in zip(self.analysis_pipeline, self.epsilons):
                clustering_function = \
                    lambda sim: DBSCAN(eps=epsilon, min_samples=self.min_samples, metric="precomputed").fit(1 - sim)
                clusters = analysis.generate_clusters(clustering_function, clusters)
                layers.append(clusters)
            if prev_clusters is not None and clusters == prev_clusters:
                break
            else:
                prev_clusters = clusters
        return layers

    def cluster(self):
        if self.strategy == "iterative":
            return self.iterative_clustering()
