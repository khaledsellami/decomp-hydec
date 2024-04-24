import json
import os
from typing import Dict, List, Callable, Union

import numpy as np
import pandas as pd

from .analysis.abstractAnalysis import AbstractAnalysis
from .analysis.dependencyAnalysis import DependencyAnalysis
from .analysis.sumAnalysis import SumAnalysis
from .analysis.tfidfAnalysis import TfidfAnalysis
from .clients.parsingClient import ParsingClient
from .clients.decParsingClient import DecParsingClient


class DataHandler:
    SUPPORTED_ANALYSIS = ["dynamic", "structural", "semantic", "sum"]
    DATA_PATH = os.path.join(os.curdir, "data")
    DEFAULTS = {
        "dynamic": {"features": "{}_calls.npy", "atoms": "{}_names.json"},
        "structural": {"features": "{}_calls.npy", "atoms": "{}_names.json"},
        "semantic": {"features": "{}_tfidf.npy", "atoms": "{}_names.json", "tokens": "{}_words.json"},
    }

    def __init__(self, app_name: str, app_repo: str = "", granularity: str = "class", is_distributed: bool = False,
                 use_module: bool = True, data_path: Union[str, None] = None, *args, **kwargs):
        assert granularity in ["class", "method"]
        self.app_name = app_name
        self.app_repo = app_repo
        self.data_path = data_path if data_path is not None else os.path.join(self.DATA_PATH, self.app_name.lower())
        self.granularity = granularity
        if use_module:
            try:
                import decparsing
                self.parsing_client = DecParsingClient(app_name, app_repo, granularity=granularity,
                                                       is_distributed=is_distributed, *args, **kwargs)
            except ImportError:
                print("Warning: decparsing module not found, using the parsing grpc client instead!")
                self.parsing_client = ParsingClient(app_name, app_repo, granularity=granularity,
                                                    is_distributed=is_distributed, *args, **kwargs)
        else:
            self.parsing_client = ParsingClient(app_name, app_repo, granularity=granularity,
                                                is_distributed=is_distributed, *args, **kwargs)

    def load_dyn_analysis(self, hyperparams: Dict, all_atoms: List[str]) -> AbstractAnalysis:
        # TODO add support for method level analysis in local data
        if "features_path" in hyperparams and "atoms_path" in hyperparams:
            with open(hyperparams["atoms_path"], "r") as f:
                dyn_classes = json.load(f)
            features = np.load(hyperparams["features_path"])
        elif os.path.exists(os.path.join(self.data_path, "dynamic")):
            local_features = os.path.join(
                self.data_path, "dynamic", self.DEFAULTS["dynamic"]["features"].format(
                    self.granularity))
            local_atoms = os.path.join(
                self.data_path, "dynamic", self.DEFAULTS["dynamic"]["atoms"].format(
                    self.granularity))
            with open(local_atoms, "r") as f:
                dyn_classes = json.load(f)
            features = np.load(local_features)
        else:
            raise ValueError("Dynamic analysis data not found in input nor locally!")
        dynamic_analysis = DependencyAnalysis(features, all_atoms, dyn_classes, similarity=hyperparams["similarity"])
        return dynamic_analysis

    def load_str_analysis(self, hyperparams: Dict, all_atoms: List[str]) -> AbstractAnalysis:
        # TODO add support for method level analysis in local data
        if "features_path" in hyperparams and "atoms_path" in hyperparams:
            with open(hyperparams["atoms_path"], "r") as f:
                str_classes = json.load(f)
            features = np.load(hyperparams["features_path"])
        elif os.path.exists(os.path.join(self.data_path, "structural")):
            local_features = os.path.join(
                self.data_path, "structural", self.DEFAULTS["structural"]["features"].format(
                    self.granularity))
            local_atoms = os.path.join(
                self.data_path, "structural", self.DEFAULTS["structural"]["atoms"].format(
                    self.granularity))
            with open(local_atoms, "r") as f:
                str_classes = json.load(f)
            features = np.load(local_features)
        else:
            try:
                str_classes = self.parsing_client.get_names()
                features = self.parsing_client.get_calls()
            except Exception as e:
                raise ValueError("Structural analysis data not found anywhere!")
        structural_analysis = DependencyAnalysis(features, all_atoms, str_classes, similarity=hyperparams["similarity"])
        return structural_analysis

    def load_sem_analysis(self, hyperparams: Dict, all_atoms: List[str]) -> AbstractAnalysis:
        # TODO add support for method level analysis in local data
        if "features_path" in hyperparams and "atoms_path" in hyperparams:
            with open(hyperparams["atoms_path"], "r") as f:
                sem_classes = json.load(f)
            # with open(hyperparams["atoms_tokens"], "r") as f:
            #     class_tokens = json.load(f)
            features = np.load(hyperparams["features_path"])
        elif os.path.exists(os.path.join(self.data_path, "semantic")):
            local_features = os.path.join(
                self.data_path, "semantic", self.DEFAULTS["semantic"]["features"].format(
                    self.granularity))
            local_atoms = os.path.join(
                self.data_path, "semantic", self.DEFAULTS["semantic"]["atoms"].format(
                    self.granularity))
            # local_tokens = os.path.join(
            #     self.data_path, "semantic", self.DEFAULTS["semantic"]["tokens"])
            with open(local_atoms, "r") as f:
                sem_classes = json.load(f)
            features = np.load(local_features)
            # with open(local_tokens, "r") as f:
            #     class_tokens = json.load(f)
        else:
            try:
                sem_classes = self.parsing_client.get_names()
                features = self.parsing_client.get_tfidf()
            except Exception as e:
                raise ValueError("Semantic analysis data not found anywhere!")
        semantic_analysis = TfidfAnalysis(features, all_atoms, sem_classes, aggregation=hyperparams["aggregation"])
        return semantic_analysis

    def load_sum_analysis(self, hyperparams: Dict, all_atoms: List[str]) -> AbstractAnalysis:
        comb_analysis = list()
        # TODO fix
        if "weights" in hyperparams:
            weights = hyperparams["weights"]
        else:
            weights = None
        for analysis_name, hps in hyperparams["analysis"].items():
            analysis_func = self.route_analysis(analysis_name)
            comb_analysis.append(analysis_func(hps, all_atoms))
        static_analysis = SumAnalysis(comb_analysis, weights)
        return static_analysis

    def route_analysis(self, analysis_name : str) -> Callable:
        if analysis_name == "dynamic":
            analysis_func = self.load_dyn_analysis
        elif analysis_name == "structural":
            analysis_func = self.load_str_analysis
        elif analysis_name == "semantic":
            analysis_func = self.load_sem_analysis
        elif analysis_name == "sum":
            analysis_func = self.load_sum_analysis
        else:
            raise ValueError("Unrecognized analysis approach '{}'!".format(analysis_name))
        return analysis_func

    def get_atoms(self, hyperparams: Dict) -> List[str]:
        if "atoms" in hyperparams:
            atoms = hyperparams["atoms"]
        elif os.path.exists(os.path.join(self.data_path, "semantic")):
            local_atoms = os.path.join(
                self.data_path, "semantic",
                self.DEFAULTS["semantic"]["atoms"].format(self.granularity)
            )
            with open(local_atoms, "r") as f:
                atoms = json.load(f)
        else:
            try:
                atoms = self.parsing_client.get_names()
            except Exception as e:
                print(e)
                raise ValueError("Failed to retrieve the list of classes or methods!")
        return atoms


def save_data(data: pd.DataFrame, save_path: str, analysis_type: str = "dynamic") -> str:
    if analysis_type == "structural" or analysis_type == "dynamic":
        names = data.columns.tolist()
        features = data.values
        path = os.path.join(save_path, analysis_type)
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, DataHandler.DEFAULTS[analysis_type]["features"]), features)
        with open(os.path.join(path, DataHandler.DEFAULTS[analysis_type]["atoms"]), "w") as f:
            json.dump(names, f)
    elif analysis_type == "semantic":
        names = data.index.values.tolist()
        features = data.values
        path = os.path.join(save_path, analysis_type)
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, DataHandler.DEFAULTS[analysis_type]["features"]), features)
        with open(os.path.join(path, DataHandler.DEFAULTS[analysis_type]["atoms"]), "w") as f:
            json.dump(names, f)
    else:
        raise ValueError(f"Unknown analysis type {analysis_type}!")
    return path

