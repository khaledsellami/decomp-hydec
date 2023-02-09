from typing import Tuple, List

from sklearn.preprocessing import OneHotEncoder
import numpy as np


def process_outliers(microservices: np.ndarray,
                     features_list: List[np.ndarray] = None,
                     exclude_outliers: bool = True,
                     both_axes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Excludes outlier classes/methods from list of microservices and related features and increments the index otherwise.
    :param microservices: microservices index for each class/method (List[int])
    :param features_list: features that are used for evaluation
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :param both_axes: True if features where both axes represent the classes/methods
    :return: processed microservices array and features array
    """
    if exclude_outliers:
        include = microservices != -1
        if features_list is not None and len(features_list) > 0:
            if both_axes:
                features_list = [features[include][:, include] for features in features_list]
            else:
                features_list = [features[include] for features in features_list]
        microservices = microservices[include]
    else:
        x = microservices == -1
        microservices[x] = microservices.max() + 1 + np.arange(x.sum())
    return microservices, features_list


def vectorized_modularity(microservices: np.ndarray,
                          features: np.ndarray,
                          exclude_outliers: bool = True) -> float:
    """
    One hot encodes the input microservice array and measures the modularity.
    :param microservices: microservices index for each class/method (List[int])
    :param features: features that are used for evaluation
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: modularity value
    """
    # TODO: Optimize further
    microservices, f = process_outliers(microservices, [features], exclude_outliers)
    features = f[0]
    n_microservices = len(np.unique(microservices))
    if n_microservices < 2:
        return 0
    microservices = OneHotEncoder().fit_transform(microservices.reshape(-1, 1)).toarray()
    return preprocessed_modularity(microservices, features)


def preprocessed_modularity(microservices: np.ndarray, features: np.ndarray) -> float:
    """
    Calculate the modular quality for the given decomposition and features.
    :param microservices: microservices index for each class/method (one-hot encoded)
    :param features: features that are used for evaluation
    :return: modularity value
    """
    n_microservices = microservices.shape[1]
    if n_microservices < 2:
        return 0
    features = features > 0
    element_number = microservices.sum(axis=0).reshape(-1, 1)
    element_number = element_number.dot(element_number.transpose())
    mod_matrix = microservices.transpose().dot(features).dot(microservices) / element_number
    cohesion = mod_matrix.diagonal().sum()
    coupling = mod_matrix.sum() - cohesion
    return cohesion / n_microservices - coupling / (n_microservices * (n_microservices - 1))


def non_extreme_distribution(microservices: np.ndarray, s_min: int = 5, s_max: int = 19,
                             exclude_outliers: bool = True) -> float:
    """
    Process outliers and calculate the Non-Extreme Distribution of the given decomposition.
    :param microservices: microservices index for each class/method (List[int])
    :param s_min: minimum threshold
    :param s_max: maximum threshold
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: Non-Extreme Distribution value
    """
    microservices, _ = process_outliers(microservices, [], exclude_outliers)
    return preprocessed_non_extreme_distribution(microservices, s_min, s_max)


def preprocessed_non_extreme_distribution(microservices: np.ndarray, s_min: int = 5, s_max: int = 19) -> float:
    """
    Calculate the Non-Extreme Distribution of the given decomposition.
    :param microservices: microservices index for each class/method (List[int])
    :param s_min: minimum threshold
    :param s_max: maximum threshold
    :return: Non-Extreme Distribution value
    """
    unique, counts = np.unique(microservices, return_counts=True)
    n_microservices = len(unique)
    if n_microservices < 1:
        return 1
    non_extreme = ((counts >= s_min)*(counts <= s_max)).sum()
    ned = 1 - non_extreme / n_microservices
    return ned


def coverage(microservices: np.ndarray) -> float:
    """
    Calculate the class/method coverage in the decomposition
    :param microservices: microservices index for each class/method (List[int]) where outliers are referred by -1
    :return: coverage value
    """
    return (microservices != -1).sum() / len(microservices)


def inter_call_percentage(microservices: np.ndarray, interactions: np.ndarray,
                          exclude_outliers: bool = True) -> float:
    """
    One hot encode the input microservice array and measure the inter-call percentage.
    :param microservices: microservices index for each class/method (List[int])
    :param interactions: class/method interaction matrix
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: inter-call percentage value
    """
    microservices, f = process_outliers(microservices, [interactions], exclude_outliers)
    interactions = f[0]
    n_microservices = len(np.unique(microservices))
    if n_microservices < 1:
        return 1
    microservices = OneHotEncoder().fit_transform(microservices.reshape(-1, 1)).toarray()
    return preprocessed_inter_call_percentage(microservices, interactions)


def preprocessed_inter_call_percentage(microservices: np.ndarray, interactions: np.ndarray) -> float:
    """
    Calculate the inter-call percentage.
    :param microservices: microservices index for each class/method (one-hot encoded)
    :param interactions: class/method interaction matrix
    :return: inter-call percentage value
    """
    n_microservices = microservices.shape[1]
    if n_microservices < 1:
        return 1
    interactions += 1
    total = np.log(interactions)
    total = microservices.transpose().dot(total).dot(microservices)
    inter = total.sum() - total.diagonal().sum()
    total = total.sum()
    if total == 0:
        return 0
    else:
        return inter/total


def interface_number(microservices: np.ndarray, interactions: np.ndarray,
                     exclude_outliers: bool = True) -> float:
    """
    One hot encode the input microservice array and measure the number of interface classes/methods.
    :param microservices: microservices index for each class/method (List[int])
    :param interactions: class/method interaction matrix
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: Interface number
    """
    microservices, f = process_outliers(microservices, [interactions], exclude_outliers)
    interactions = f[0]
    n_microservices = len(np.unique(microservices))
    if n_microservices < 1:
        return interactions.shape[0]
    microservices = OneHotEncoder().fit_transform(microservices.reshape(-1, 1)).toarray()
    return preprocessed_interface_number(microservices, interactions)


def preprocessed_interface_number(microservices: np.ndarray, interactions: np.ndarray) -> float:
    """
    Calculate the number of interface classes/methods.
    :param microservices: microservices index for each class/method (one-hot encoded)
    :param interactions: class/method interaction matrix
    :return: Interface number
    """
    n_microservices = microservices.shape[1]
    if n_microservices < 1:
        return interactions.shape[0]
    interfaces = microservices.transpose().dot(interactions)
    interfaces = interfaces * (1 - microservices.transpose())
    return (interfaces.sum(0) > 0).sum() / n_microservices


def microservices_number(microservices: np.ndarray, exclude_outliers: bool = True) -> int:
    """
    Calculate the number of microservices.
    :param microservices: microservices index for each class/method (List[int])
    :param exclude_outliers: True to exclude outliers (denoted by -1)
    :return: number of microservices in the decomposition
    """
    n_micro = len(np.unique(microservices)) - np.any(microservices == -1)
    if not exclude_outliers:
        n_micro += np.sum(microservices == -1)
    return n_micro


def precision_and_recall(truth_ind, inferred_ind, v2=True):
    truth_clusters = np.array([truth_ind==i for i in np.unique(truth_ind)]).astype(int)
    inferred_clusters = np.array([inferred_ind==i for i in np.unique(inferred_ind)]).astype(int)
    if v2:
        df = truth_clusters.dot(inferred_clusters.transpose())
        inter = df/inferred_clusters.sum(axis=1)
        correspondance = ((inter==inter.max(axis=0))*(1/truth_clusters.sum(axis=1).transpose().reshape(-1,1))).argmax(axis=0)
        precision = 0
        recall = 0
        for i,j in enumerate(correspondance):
            precision += df[j,i]/inferred_clusters[i].sum()
            recall += df[j,i]/truth_clusters[j].sum()
        n = len(inferred_clusters)
        return precision/n, recall/n
    else:
        df = inferred_clusters.dot(truth_clusters.transpose())
        correspondance = (df/truth_clusters.sum(axis=1)).argmax(axis=0)
        precision = 0
        recall = 0
        for i,j in enumerate(correspondance):
            precision += df[j,i]/inferred_clusters[j].sum()
            recall += df[j,i]/truth_clusters[i].sum()
        n = max(len(correspondance), len(inferred_clusters))
        return precision/n, recall/n


def success_rate(truth_ind, inferred_ind, threshold):
    max_c = max(truth_ind.max(),inferred_ind.max())
    truth_clusters = np.array([truth_ind==i for i in range(1, truth_ind.max()+1)]).astype(int)
    inferred_clusters = np.array([inferred_ind==i for i in range(1, inferred_ind.max()+1)]).astype(int)
    df = truth_clusters.dot(inferred_clusters.transpose())
    inter = df/inferred_clusters.sum(axis=1)
    correspondance = ((inter==inter.max(axis=0))*(1/truth_clusters.sum(axis=1).transpose().reshape(-1,1))).argmax(axis=0)
    success_rate1, success_rate2 = 0, 0
    for i,j in enumerate(correspondance):
        success_rate1 += 1 if df[j,i]/inferred_clusters[i].sum()>=threshold else 0
        success_rate2 += 1 if df[j,i]/truth_clusters[j].sum()>=threshold else 0
    n = len(inferred_clusters)
    return success_rate1/n, success_rate2/n