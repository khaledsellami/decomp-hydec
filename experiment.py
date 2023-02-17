import datetime
import os
import json
import time
import logging
from typing import Dict, Union, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from analysis.abstractAnalysis import AbstractAnalysis
from analysis.bertAnalysis import BertAnalysis
from analysis.dependencyAnalysis import DependencyAnalysis
from analysis.sumAnalysis import SumAnalysis
from analysis.tfidfAnalysis import TfidfAnalysis
from evaluation.evaluationHandler import EvaluationHandler
from hybridDecomp import HybridDecomp
from analysis.wordEmbeddingAnalysis import WordEmbeddingAnalysis
from analysis.embeddingModel import EmbeddingModel
from utils.logTraceParser import LogTraceParcer


def load_word_embed_analysis(hyperparams: Dict) -> AbstractAnalysis:
    with open(hyperparams["atoms_path"], "r") as f:
        sem_classes = json.load(f)
    with open(hyperparams["features_path"], "r") as f:
        class_text = [c["textAndNames"] for c in json.load(f)]
    embedding_model = EmbeddingModel(**hyperparams["model"])
    embedding_model.load_embedding_model()
    # class_tokens = [embedding_model.tokenize(text) for text in class_text]
    emb_analysis = WordEmbeddingAnalysis(class_text, embedding_model, sem_classes, sem_classes,
                                         aggregation=hyperparams["aggregation"])
    return emb_analysis


def load_dyn_analysis(hyperparams: Dict) -> AbstractAnalysis:
    with open(hyperparams["atoms_path"], "r") as f:
        sem_classes = json.load(f)
    dynamic_analysis_data = hyperparams["features_path"]
    dyn_analysis = LogTraceParcer(dynamic_analysis_data)
    short_names = [c.split(".")[-1] for c in sem_classes]
    filtered_classes = [i for i in dyn_analysis.class_names if i.split("::")[-1] not in short_names]
    dyn_analysis = LogTraceParcer(dynamic_analysis_data, filtered_classes)
    dynamic_analysis = DependencyAnalysis(dyn_analysis.class_relations, short_names,
                                          [i.split("::")[-1] for i in dyn_analysis.class_names],
                                          similarity=hyperparams["similarity"])
    return dynamic_analysis


def load_str_analysis(hyperparams: Dict) -> AbstractAnalysis:
    with open(hyperparams["atoms_path"], "r") as f:
        str_classes = json.load(f)
    features = np.load(hyperparams["features_path"])
    structural_analysis = DependencyAnalysis(features, str_classes, str_classes, similarity=hyperparams["similarity"])
    return structural_analysis


def load_sem_analysis(hyperparams: Dict) -> AbstractAnalysis:
    with open(hyperparams["atoms_path"], "r") as f:
        sem_classes = json.load(f)
    features = np.load(hyperparams["features_path"])
    with open(hyperparams["atoms_tokens"], "r") as f:
        class_tokens = json.load(f)
    semantic_analysis = TfidfAnalysis(features, sem_classes, sem_classes, class_tokens,
                                      aggregation=hyperparams["aggregation"])
    return semantic_analysis


def load_bert_analysis(hyperparams: Dict, save_embeddings: bool = False) -> AbstractAnalysis:
    with open(hyperparams["atoms_path"], "r") as f:
        sem_classes = json.load(f)
    with open(hyperparams["features_path"], "r") as f:
        features = json.load(f)
    if not save_embeddings and "embeddings_path" in hyperparams:
        embeddings = np.load(hyperparams["embeddings_path"])
    else:
        embeddings = None
    bert_analysis = BertAnalysis(features, sem_classes, sem_classes, aggregation=hyperparams["aggregation"],
                                 model_name=hyperparams["model_name"], embeddings=embeddings)
    if save_embeddings and "embeddings_path" in hyperparams:
        np.save(hyperparams["embeddings_path"], bert_analysis.embeddings)
    return bert_analysis


def load_sum_analysis(hyperparams: Dict) -> AbstractAnalysis:
    comb_analysis = list()
    if "weights" in hyperparams:
        weights = hyperparams["weights"]
    else:
        weights = None
    for analysis_name, hps in hyperparams["analysis"]:
        if analysis_name in ANALYSIS_MAP:
            comb_analysis.append(ANALYSIS_MAP[analysis_name](hps))
        else:
            raise ValueError("Unrecognized analysis approach '{}'!".format(analysis_name))
    static_analysis = SumAnalysis(comb_analysis, weights)
    return static_analysis


ANALYSIS_MAP = {
    "word_embedding": load_word_embed_analysis,
    "dynamic": load_dyn_analysis,
    "structural": load_str_analysis,
    "semantic": load_sem_analysis,
    "bert_embedding": load_bert_analysis,
    "sum": load_sum_analysis
}


class Experiment:
    def __init__(self, app: str, hyperparams: Dict):
        self.app = app
        self.hyperparams = hyperparams
        start_date = datetime.datetime.now()
        self.experiment_id = "hyDec_{}_{}".format(self.app.lower(), start_date.strftime("%Y%m%d%H%M%S"))
        self.output_path = os.path.join(os.path.curdir, "logs", self.app.lower(), self.experiment_id)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        self.experiment_metadata = dict()
        self.experiment_metadata["experiment_id"] = self.experiment_id
        self.experiment_metadata["application"] = self.app
        self.experiment_metadata["start_datetime"] = start_date.strftime("%Y-%m-%d %H:%M:%S")
        self.experiment_metadata["hyperparameters"] = dict()
        self.experiment_metadata["exec_time"] = dict()
        self.analysis_pipeline = list()
        self.epsilons = list()
        self.logger = logging.getLogger('Experiment')

    def run(self):
        t1 = time.time()
        self.logger.debug("Starting decomposition process")
        for analysis_name, hps in self.hyperparams["analysis"]:
            if analysis_name in ANALYSIS_MAP:
                self.load_analysis(analysis_name, hps)
            else:
                raise ValueError("Unrecognized analysis approach '{}'!".format(analysis_name))
        # eval
        eval_handler = self.init_evaluation(self.hyperparams["evaluation"])
        # cluster
        layers, atom_list = self.cluster(self.hyperparams["clustering"])
        exec_time = time.time() - t1
        self.experiment_metadata["exec_time"]["total"] = exec_time
        # save results
        self.eval_results(layers, eval_handler, atom_list)
        self.logger.debug("Finished decomposition process")

    def init_evaluation(self, hyperparams: Dict) -> EvaluationHandler:
        t1 = time.time()
        self.logger.debug("Initializing evaluation")
        self.experiment_metadata["evaluation_hps"] = hyperparams.copy()
        with open(hyperparams["semantic_data_path"], "r") as f:
            class_words = json.load(f)
        semantic_data = CountVectorizer().fit_transform([" ".join(i) for i in class_words]).toarray()
        semantic_data = semantic_data[:, semantic_data.sum(axis=0) < hyperparams["threshold"]]
        semantic_data = semantic_data.dot(semantic_data.transpose())
        eval_handler = EvaluationHandler(hyperparams["structural_data_path"], semantic_data,
                                         metrics=hyperparams["metrics"])
        exec_time = time.time() - t1
        self.experiment_metadata["exec_time"]["evaluation_init"] = exec_time
        self.logger.debug("Finished initializing evaluation ({:.4f}s)".format(exec_time))
        return eval_handler

    def cluster(self, hyperparams: Dict) -> Tuple[Union[List[np.ndarray], List[List[int]]], List]:
        t1 = time.time()
        self.logger.debug("Starting clustering")
        self.experiment_metadata["clustering_hps"] = hyperparams.copy()
        with open(hyperparams["atoms_path"], "r") as f:
            sem_classes = json.load(f)
        hybrid_decomp = HybridDecomp(self.analysis_pipeline, sem_classes, self.epsilons,
                                     strategy=hyperparams["strategy"], max_iterations=hyperparams["max_iterations"],
                                     min_samples=hyperparams["min_samples"], epsilon_step=hyperparams["epsilon_step"],
                                     include_outliers=hyperparams["include_outliers"]
                                     )
        layers = hybrid_decomp.cluster()
        exec_time = time.time() - t1
        self.experiment_metadata["exec_time"]["clustering"] = exec_time
        self.logger.debug("Finished clustering ({:.4f}s)".format(exec_time))
        return layers, hybrid_decomp.atoms

    def eval_results(self, layers: Union[List[np.ndarray], List[List[int]]], eval_handler: EvaluationHandler,
                     atom_list: List, log: bool = True, save: bool = True):
        t1 = time.time()
        self.logger.debug("Evaluating results")
        metrics = eval_handler.metrics
        if log:
            self.logger.info("metrics: {}".format(metrics))
        all_results = list()
        for i, layer in enumerate(layers):
            results = eval_handler.evaluate(layer)
            all_results.append([self.experiment_id, i] + list(results.values()))
            if log:
                self.logger.info("{}_{}_{}".format(i, layer, results.values()))
        exec_time = time.time() - t1
        self.experiment_metadata["exec_time"]["evaluation_calc"] = exec_time
        self.logger.debug("Finished evaluating ({:.4f}s)".format(exec_time))
        if save:
            t1 = time.time()
            self.logger.debug("Saving results")
            df = pd.DataFrame(all_results, columns=["experiment_id", "layer_id"] + metrics)
            df.to_csv(os.path.join(self.output_path, "results.csv".format(self.experiment_id)), index=False)
            df_layers = pd.DataFrame(layers, columns=atom_list)
            df_layers.to_csv(os.path.join(self.output_path, "layers.csv"), index=False)
            with open(os.path.join(self.output_path, "experiment_metadata.json"), "w") as f:
                json.dump(self.experiment_metadata, f, indent=4)
            exec_time = time.time() - t1
            self.logger.debug("Finished saving results ({:.4f}s)".format(exec_time))

    def load_analysis(self, analysis_name: str, hyperparams: Dict, save: bool = True):
        assert analysis_name in ANALYSIS_MAP
        t1 = time.time()
        self.logger.debug("Initializing {} analysis".format(analysis_name))
        if save:
            self.experiment_metadata["hyperparameters"][analysis_name] = hyperparams.copy()
        analysis_obj = ANALYSIS_MAP[analysis_name](hyperparams)
        self.analysis_pipeline.append(analysis_obj)
        self.epsilons.append(hyperparams["epsilon"])
        exec_time = time.time() - t1
        if save:
            self.experiment_metadata["exec_time"][analysis_name] = exec_time
        self.logger.debug("Finished {} ({:.4f}s)".format(analysis_name, exec_time))

