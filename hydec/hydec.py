import json
import os.path
from typing import Dict, Tuple, Union, List

import numpy as np

from .experiment import Experiment
from .dataHandler import DataHandler
from .utils.default_hyperparams import generate_hps_input, get_default_hyperparams, merge_input


APPROACH_PIPELINES = {
    "hyDec":["dynamic", ["structural", "semantic"]],
    "hierDec":[["structural", "semantic"]]
}


def generate_decomposition(app: str, app_repo: str = None, decomp_approach: str = "hyDec", hyperparams_path: str = None,
                           structural_path: str = None, semantic_path: str = None, dynamic_path: str = None,
                           include_metadata: bool = False, save_output: bool = False, granularity: str = "class",
                           is_distributed: bool = False, use_parsing_module: bool = True,
                           data_path: Union[str, None] = None, *args, **kwargs
                           ) -> Tuple[Union[np.ndarray, List[int]], Union[List[np.ndarray], List[List[int]]], List,
                                      Union[None, Dict]]:
    assert granularity in ["class", "method"]
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
            structural_path, DataHandler.DEFAULTS["structural"]["atoms"].format(granularity))
        hyperparams["structural"]["features_path"] = os.path.join(
            structural_path, DataHandler.DEFAULTS["structural"]["features"].format(granularity))
    if dynamic_path is not None:
        hyperparams["dynamic"]["atoms_path"] = os.path.join(
            dynamic_path, DataHandler.DEFAULTS["dynamic"]["atoms"].format(granularity))
        hyperparams["dynamic"]["features_path"] = os.path.join(
            dynamic_path, DataHandler.DEFAULTS["dynamic"]["features"].format(granularity))
    if semantic_path is not None:
        hyperparams["semantic"]["atoms_path"] = os.path.join(
            semantic_path, DataHandler.DEFAULTS["semantic"]["atoms"].format(granularity))
        hyperparams["semantic"]["features_path"] = os.path.join(
            semantic_path, DataHandler.DEFAULTS["semantic"]["features"].format(granularity))
        # hyperparams["semantic"]["atoms_tokens"] = os.path.join(
        #     structural_path, DataHandler.DEFAULTS["semantic"]["tokens"])
    hp_input = generate_hps_input(analysis_pipeline, hyperparams)
    hp_input["clustering"] = hyperparams["clustering"]
    hps = {"clustering": hp_input["clustering"],
           "analysis": [(i, j) for i, j in hp_input.items() if i not in ["clustering"]]}
    experiment = Experiment(app, hps, app_repo=app_repo, decomp_approach=decomp_approach,
                            include_metadata=include_metadata, save_output=save_output, granularity=granularity,
                            is_distributed=is_distributed, use_parsing_module=use_parsing_module,
                            data_path=data_path, *args, **kwargs)
    layers, atoms, metadata = experiment.run()
    return layers[-1], layers, atoms, metadata