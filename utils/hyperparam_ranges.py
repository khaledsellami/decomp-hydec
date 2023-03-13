import os
from typing import List, Dict

import numpy as np

from user_config import DYN_DATA_PATH


def get_hyperparams(names: List[str], data_path: str, app: str) -> Dict[str, Dict]:
    if DYN_DATA_PATH is None and "dynamic" in names:
        raise FileNotFoundError("Dynamic data was not found!")
    sem_classes = os.path.join(data_path, app.lower(), "semantic_data", "class_names.json")
    str_classes = os.path.join(data_path, app.lower(), "structural_data", "class_names.json")

    hp_ranges = dict()

    type_data = os.path.join(data_path, app.lower(), "static_analysis_results", "typeData.json")
    hp_ranges["word_embedding"] = dict(
       atoms_path=sem_classes,
       features_path=type_data,
       epsilon=np.arange(0.05, 1, 0.01),
       model=dict(
           type="fasttext",
           pooling_approach="avg",  # ["avg", "max"],
           dim=300
       ),
       fine_tuned=False,
       aggregation=["mean", "sum", "combine"]
    )
    hp_ranges["dynamic"] = dict(
       atoms_path=sem_classes,
       features_path=DYN_DATA_PATH,
       epsilon=np.arange(0.05, 1, 0.1),
       similarity=["call", "cousage"]
   )
    class_calls = os.path.join(data_path, app.lower(), "structural_data", "class_calls.npy")
    hp_ranges["structural"] = dict(
       atoms_path=str_classes,
       features_path=class_calls,
       epsilon=np.arange(0.05, 1, 0.1),
       similarity=["call", "cousage"]
   )

    class_tfidf = os.path.join(data_path, app.lower(), "semantic_data", "class_tfidf.npy")
    class_words = os.path.join(data_path, app.lower(), "semantic_data", "class_words.json")
    hp_ranges["semantic"] = dict(
        atoms_path=sem_classes,
        features_path=class_tfidf,
        atoms_tokens=class_words,
        epsilon=np.arange(0.05, 1, 0.1),
        aggregation=["mean", "sum"]
    )

    embeddings = os.path.join(data_path, app.lower(), "embeddings.npy")
    hp_ranges["bert_embedding"] = dict(
        atoms_path=sem_classes,
        features_path=type_data,
        embeddings_path=embeddings,
        epsilon=np.arange(0.05, 0.9, 0.05),
        aggregation=["mean", "sum"],
        model_name="bert-base-uncased"
    )

    class_interactions = os.path.join(data_path, app.lower(), "structural_data", "interactions.npy")
    hp_ranges["evaluation"] = dict(
        structural_data_path=class_interactions,
        semantic_data_path=class_words,
        metrics=["smq", "cmq", "icp", "ifn", "ned", "cov", "msn", "comb"],
        threshold=50
    )

    hp_ranges["clustering"] = dict(
        atoms_path=sem_classes,
        strategy=["alternating", "alternating_epsilon", "sequential"],
        max_iterations=100,
        min_samples=[2, 3],
        epsilon_step=[0.01, 0.05, 0.1],
        include_outliers=False
    )

    return {k: v for k, v in hp_ranges.items() if k in names}
