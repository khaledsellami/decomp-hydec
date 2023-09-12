import json
import os.path
from typing import Dict, Tuple, Union, List

import numpy as np

from experiment import Experiment
from dataHandler import DataHandler
from utils.default_hyperparams import generate_hps_input, get_default_hyperparams, merge_input


APPROACH_PIPELINES = {
    "hyDec":["dynamic", ["structural", "semantic"]],
    "hierDec":[["structural", "semantic"]]
}


def generate_decomposition(app: str, app_repo: str = None, decomp_approach: str = "hyDec", hyperparams_path: str = None,
                           structural_path: str = None, semantic_path: str = None, dynamic_path: str = None,
                           include_metadata: bool = False, save_output: bool = False
                           ) -> Tuple[Union[np.ndarray, List[int]], Union[List[np.ndarray], List[List[int]]], List,
                                      Union[None, Dict]]:
    analysis_pipeline = APPROACH_PIPELINES[decomp_approach]
    if hyperparams_path is not None:
        default_hps = get_default_hyperparams()
        with open(hyperparams_path, "r") as f:
            hyperparams = json.load(f)
        hyperparams = merge_input(hyperparams, default_hps)
    else:
        default_hps = get_default_hyperparams()
        hyperparams = default_hps
    if structural_path is not None:
        hyperparams["structural"]["atoms_path"] = os.path.join(
            structural_path, DataHandler.DEFAULTS["structural"]["atoms"])
        hyperparams["structural"]["features_path"] = os.path.join(
            structural_path, DataHandler.DEFAULTS["structural"]["features"])
    if dynamic_path is not None:
        hyperparams["dynamic"]["atoms_path"] = os.path.join(
            dynamic_path, DataHandler.DEFAULTS["dynamic"]["atoms"])
        hyperparams["dynamic"]["features_path"] = os.path.join(
            dynamic_path, DataHandler.DEFAULTS["dynamic"]["features"])
    if semantic_path is not None:
        hyperparams["semantic"]["atoms_path"] = os.path.join(
            semantic_path, DataHandler.DEFAULTS["semantic"]["atoms"])
        hyperparams["semantic"]["features_path"] = os.path.join(
            semantic_path, DataHandler.DEFAULTS["semantic"]["features"])
        # hyperparams["semantic"]["atoms_tokens"] = os.path.join(
        #     structural_path, DataHandler.DEFAULTS["semantic"]["tokens"])
    hp_input = generate_hps_input(analysis_pipeline, hyperparams)
    hp_input["clustering"] = hyperparams["clustering"]
    hps = {"clustering": hp_input["clustering"],
           "analysis": [(i, j) for i, j in hp_input.items() if i not in ["clustering"]]}
    experiment = Experiment(app, hps, app_repo=app_repo, decomp_approach=decomp_approach,
                            include_metadata=include_metadata, save_output=save_output)
    layers, atoms, metadata = experiment.run()
    return layers[-1], layers, atoms, metadata