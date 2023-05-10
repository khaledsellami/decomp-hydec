import os
from typing import List, Dict, Union

import numpy as np

from user_config import DYN_DATA_PATH


def get_hyperparams(names: List[Union[str, List[str]]], data_path: str, app: str) -> Dict[str, Dict]:
    sem_classes = os.path.join(data_path, app.lower(), "semantic_data", "class_names.json")
    str_classes = os.path.join(data_path, app.lower(), "structural_data", "class_names.json")
    dyn_classes = os.path.join(data_path, app.lower(), "dynamic_analysis", "class_names.json")

    hp_ranges = dict()

    type_data = os.path.join(data_path, app.lower(), "static_analysis_results", "typeData.json")
    hp_ranges["word_embedding"] = dict(
       atoms_path=sem_classes,
       features_path=type_data,
       epsilon=list(np.arange(0.05, 1, 0.01)),
       model=dict(
           type="fasttext",
           pooling_approach="avg",  # ["avg", "max"],
           dim=300
       ),
       fine_tuned=False,
       aggregation=["mean", "sum", "combine"]
    )
    dyn_class_calls = os.path.join(data_path, app.lower(), "dynamic_analysis", "class_calls.npy")
    hp_ranges["dynamic"] = dict(
       atoms_path=dyn_classes,
       # all_atoms_path=sem_classes,
       features_path=dyn_class_calls,
       epsilon=list(np.arange(0.05, 1, 0.1)),
       similarity=["call", "cousage"]
   )
    class_calls = os.path.join(data_path, app.lower(), "structural_data", "class_calls.npy")
    hp_ranges["structural"] = dict(
       atoms_path=str_classes,
       features_path=class_calls,
       epsilon=list(np.arange(0.05, 1, 0.1)),
       similarity=["call", "cousage"]
   )

    class_tfidf = os.path.join(data_path, app.lower(), "semantic_data", "class_tfidf.npy")
    class_words = os.path.join(data_path, app.lower(), "semantic_data", "class_words.json")
    hp_ranges["semantic"] = dict(
        atoms_path=sem_classes,
        features_path=class_tfidf,
        atoms_tokens=class_words,
        epsilon=list(np.arange(0.05, 1, 0.1)),
        aggregation=["mean", "sum"]
    )

    embeddings = os.path.join(data_path, app.lower(), "embeddings.npy")
    hp_ranges["bert_embedding"] = dict(
        atoms_path=sem_classes,
        features_path=type_data,
        embeddings_path=None,# embeddings,
        epsilon=list(np.arange(0.01, 1, 0.01)),
        aggregation=["mean", "sum", "combine"],
        model_name="bert-base-uncased"
    )
    hp_ranges["sum"] = dict(
        analysis=None,
        weights=None,
        epsilon=list(np.arange(0.05, 1, 0.1))
    )

    class_interactions = os.path.join(data_path, app.lower(), "structural_data", "interactions.npy")
    hp_ranges["evaluation"] = dict(
        structural_data_path=class_interactions,
        semantic_data_path=class_words,
        metrics=["smq", "cmq", "icp", "ifn", "ned", "cov", "msn", "comb"],
        threshold=15
    )

    hp_ranges["clustering"] = dict(
        atoms_path=sem_classes,
        strategy=["alternating", "alternating_epsilon", "sequential"],
        max_iterations=1000,
        min_samples=[2, 3],
        epsilon_step=[0.01, 0.05, 0.1],
        include_outliers=False
    )

    def get_hps(pipeline):
        hps = dict()
        for k in pipeline:
            if isinstance(k, list):
                hps["sum"] = hp_ranges["sum"].copy()
                hps["sum"]["analysis"] = get_hps(k)
            else:
                hps[k] = hp_ranges[k]
        return hps
    return get_hps(names)
