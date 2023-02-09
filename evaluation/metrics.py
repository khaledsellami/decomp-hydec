from sklearn.preprocessing import OneHotEncoder
import numpy as np


def process_outliers(microservices, features_list, exclude_outliers=True, both_axes=True):
    if exclude_outliers:
        include = microservices != -1
        if len(features_list) > 0:
            if both_axes:
                features_list = [features[include][:, include] for features in features_list]
            else:
                features_list = [features[include] for features in features_list]
        microservices = microservices[include]
    else:
        x = microservices == -1
        microservices[x] = microservices.max() + 1 + np.arange(x.sum())
    return microservices, features_list


def vectorized_modularity(microservices, features, exclude_outliers=True):
    # TODO: Optimize further
    microservices, f = process_outliers(microservices, [features], exclude_outliers)
    features = f[0]
    n_microservices = len(np.unique(microservices))
    if n_microservices < 2:
        return 0
    microservices = OneHotEncoder().fit_transform(microservices.reshape(-1, 1)).toarray()
    return preprocessed_modularity(microservices, features)


def preprocessed_modularity(microservices, features):
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


def non_extreme_distribution(microservices, s_min=5, s_max=19, exclude_outliers=True):
    microservices, _ = process_outliers(microservices, [], exclude_outliers)
    return preprocessed_non_extreme_distribution(microservices, s_min, s_max)


def preprocessed_non_extreme_distribution(microservices, s_min=5, s_max=19):
    unique, counts = np.unique(microservices, return_counts=True)
    n_microservices = len(unique)
    if n_microservices < 1:
        return 1
    non_extreme = ((counts >= s_min)*(counts <= s_max)).sum()
    ned = 1 - non_extreme / n_microservices
    return ned


def coverage(microservices):
    return (microservices != -1).sum() / len(microservices)


def inter_call_percentage(microservices, interactions, exclude_outliers=True):
    microservices, f = process_outliers(microservices, [interactions], exclude_outliers)
    interactions = f[0]
    n_microservices = len(np.unique(microservices))
    if n_microservices < 1:
        return 1
    microservices = OneHotEncoder().fit_transform(microservices.reshape(-1, 1)).toarray()
    return preprocessed_inter_call_percentage(microservices, interactions)


def preprocessed_inter_call_percentage(microservices, interactions):
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


def interface_number(microservices, interactions, exclude_outliers=True):
    microservices, f = process_outliers(microservices, [interactions], exclude_outliers)
    interactions = f[0]
    n_microservices = len(np.unique(microservices))
    if n_microservices < 1:
        return interactions.shape[0]
    microservices = OneHotEncoder().fit_transform(microservices.reshape(-1, 1)).toarray()
    return preprocessed_interface_number(microservices, interactions)


def preprocessed_interface_number(microservices, interactions):
    n_microservices = microservices.shape[1]
    if n_microservices < 1:
        return interactions.shape[0]
    interfaces = microservices.transpose().dot(interactions)
    interfaces = interfaces * (1 - microservices.transpose())
    return (interfaces.sum(0) > 0).sum() / n_microservices


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