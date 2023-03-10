from typing import List, Union, Dict

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .metrics import process_outliers, preprocessed_modularity, preprocessed_interface_number, \
    preprocessed_inter_call_percentage, preprocessed_non_extreme_distribution, coverage, microservices_number, \
    combine_metrics, METRICS_RANGES


class EvaluationHandler:
    ALLOWED_METRICS = ["smq", "cmq", "icp", "ifn", "ned", "ned_avg", "cov", "msn", "comb"]
    DEFAULT_METRICS = ["smq", "cmq", "icp", "ifn", "ned", "cov", "msn", "comb"]

    def __init__(self, interaction_data: Union[str, np.ndarray] = None,
                 semantic_data: Union[str, np.ndarray] = None,
                 metrics: List[str] = None,
                 exclude_outliers: bool = True):
        """
        Initialize the class and load the required data.
        :param interaction_data: data required for the metrics SMQ, IFN, and ICP
        :param semantic_data: data required for the metric CMQ
        :param metrics: list of metrics to calculated
        :param exclude_outliers: True to exclude outliers in calculations (outliers are denoted by -1)
        """
        # Initialize metrics to measure
        if metrics is None:
            self.metrics = self.DEFAULT_METRICS
        else:
            if any([m not in self.ALLOWED_METRICS for m in metrics]):
                raise ValueError("Unrecognized metric in input!")
            self.metrics = metrics
        # Load interaction data if required
        if interaction_data is None and any(m in self.metrics for m in ["smq", "icp", "ifn"]):
            raise ValueError("interaction data is required for measuring SMQ, ICP and IFN")
        elif isinstance(interaction_data, str):
            self.interaction_data = np.load(interaction_data)
        elif isinstance(interaction_data, np.ndarray):
            self.interaction_data = interaction_data
        else:
            self.interaction_data = None
        # Load semantic data if required
        if semantic_data is None and "cmq" in self.metrics:
            raise ValueError("semantic data is required for measuring CMQ")
        elif isinstance(semantic_data, str):
            self.semantic_data = np.load(semantic_data)
            self.semantic_data = self.semantic_data.dot(self.semantic_data.transpose())
        elif isinstance(semantic_data, np.ndarray):
            self.semantic_data = semantic_data
        else:
            self.semantic_data = None
        # other parameters
        self.exclude_outliers = exclude_outliers

    def evaluate(self, microservices: np.ndarray) -> Dict[str, Union[int, float]]:
        """
        Calculate evaluation metrics.
        :param microservices: microservices index for each class/method (List[int])
        :return: results dictionary
        """
        if isinstance(microservices, list):
            microservices = np.array(microservices)
        results = dict()
        features_list = [f for f in [self.interaction_data, self.semantic_data] if f is not None]
        processed_microservices, features_list = process_outliers(microservices, features_list, self.exclude_outliers)
        if len(np.unique(processed_microservices)) < 1:
            microservices_encoded = np.empty(shape=(1, 0))
        else:
            microservices_encoded = OneHotEncoder().fit_transform(processed_microservices.reshape(-1, 1)).toarray()
        i = 0
        if self.interaction_data is not None:
            interaction_data = features_list[0]
            i += 1
        else:
            interaction_data = None
        if self.semantic_data is not None:
            semantic_data = features_list[i]
        else:
            semantic_data = None
        for m, f in zip(["smq", "icp", "ifn"], [preprocessed_modularity,
                                                preprocessed_inter_call_percentage,
                                                preprocessed_interface_number]):
            if m in self.metrics:
                assert interaction_data is not None
                results[m] = f(microservices_encoded, interaction_data)
        if "cmq" in self.metrics:
            assert semantic_data is not None
            results["cmq"] = preprocessed_modularity(microservices_encoded, semantic_data)
        if "ned" in self.metrics:
            results["ned"] = preprocessed_non_extreme_distribution(processed_microservices)
        if "ned_avg" in self.metrics:
            results["ned_avg"] = preprocessed_non_extreme_distribution(processed_microservices, method="avg")
        if "cov" in self.metrics:
            results["cov"] = coverage(microservices)
        if "msn" in self.metrics:
            results["msn"] = microservices_number(microservices, self.exclude_outliers)
        # needs to be last
        if "comb" in self.metrics:
            results["comb"] = combine_metrics({m: results[m] for m in results if m in METRICS_RANGES})
        return results


