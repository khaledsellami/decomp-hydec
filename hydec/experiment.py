import datetime
import json
import os
import time
import logging
import logging.config
from typing import Dict, Union, List, Tuple

import numpy as np
import pandas as pd

from .hybridDecomp import HybridDecomp
from .dataHandler import DataHandler


class Experiment:
    def __init__(self, app: str, hyperparams: Dict, app_repo: str = "", name_append: Union[str, None] = None,
                 include_metadata: bool = True, save_output: bool = False, decomp_approach: str = "hyDec",
                 granularity: str = "class", is_distributed: bool = False):
        self.app = app
        self.app_repo = app_repo
        self.hyperparams = hyperparams
        self.save_output = save_output
        start_date = datetime.datetime.now()
        self.experiment_id = "{}_{}".format(decomp_approach, self.app.lower())
        if name_append is not None:
            self.experiment_id = self.experiment_id + "_{}".format(name_append)
        self.output_path = os.path.join(os.path.curdir, "logs", self.app.lower(), self.experiment_id)
        if self.save_output:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path, exist_ok=True)
        logging.config.fileConfig(
            os.path.join(os.curdir, "utils", 'logging.conf')
        )
        self.experiment_metadata = dict()
        self.experiment_metadata["experiment_id"] = self.experiment_id
        self.experiment_metadata["application"] = self.app
        self.experiment_metadata["granularity"] = granularity
        self.experiment_metadata["app_distributed"] = is_distributed
        self.experiment_metadata["start_datetime"] = start_date.strftime("%Y-%m-%d %H:%M:%S")
        self.experiment_metadata["hyperparameters"] = dict()
        self.experiment_metadata["exec_time"] = dict()
        self.analysis_pipeline = list()
        self.epsilons = list()
        self.logger = logging.getLogger('Experiment')
        self.include_metadata = include_metadata
        self.data_handler = DataHandler(self.app, self.app_repo, granularity=granularity, is_distributed=is_distributed)
        self.atoms = self.data_handler.get_atoms(self.hyperparams["clustering"])

    def run(self) -> Tuple[Union[List[np.ndarray], List[List[int]]], List, Union[None, Dict]]:
        self.logger.info("Starting experiment {}".format(self.experiment_id))
        t1 = time.time()
        for analysis_name, hps in self.hyperparams["analysis"]:
            self.experiment_metadata["hyperparameters"][analysis_name] = hps.copy()
        self.experiment_metadata["clustering_hps"] = self.hyperparams["clustering"].copy()
        # if self.save:
        #     with open(os.path.join(self.output_path, "experiment_metadata.json"), "w") as f:
        #         json.dump(self.experiment_metadata, f, indent=4)
        self.logger.debug("Starting decomposition process")
        for analysis_name, hps in self.hyperparams["analysis"]:
            if analysis_name in self.data_handler.SUPPORTED_ANALYSIS:
                self.load_analysis(analysis_name, hps, self.atoms)
            else:
                raise ValueError("Unrecognized analysis approach '{}'!".format(analysis_name))
        # eval
        # eval_handler = self.init_evaluation(self.hyperparams["evaluation"])
        # cluster
        layers, atom_list = self.cluster(self.hyperparams["clustering"])
        exec_time = time.time() - t1
        self.experiment_metadata["exec_time"]["total"] = exec_time
        # save results
        # self.eval_results(layers, eval_handler, atom_list)
        self.logger.debug("Finished decomposition process")
        self.log_output(layers, atom_list)
        self.logger.info("Finished experiment {}".format(self.experiment_id))
        return layers, atom_list, (self.experiment_metadata if self.include_metadata else None)

    # def init_evaluation(self, hyperparams: Dict) -> EvaluationHandler:
    #     t1 = time.time()
    #     self.logger.debug("Initializing evaluation")
    #     # self.experiment_metadata["evaluation_hps"] = hyperparams.copy()
    #     with open(hyperparams["semantic_data_path"], "r") as f:
    #         class_words = json.load(f)
    #     semantic_data = CountVectorizer().fit_transform([" ".join(i) for i in class_words]).toarray()
    #     semantic_data = semantic_data[:, semantic_data.sum(axis=0) < hyperparams["threshold"]]
    #     semantic_data = semantic_data.dot(semantic_data.transpose())
    #     eval_handler = EvaluationHandler(hyperparams["structural_data_path"], semantic_data,
    #                                      metrics=hyperparams["metrics"])
    #     exec_time = time.time() - t1
    #     self.experiment_metadata["exec_time"]["evaluation_init"] = exec_time
    #     self.logger.debug("Finished initializing evaluation ({:.4f}s)".format(exec_time))
    #     return eval_handler

    def cluster(self, hyperparams: Dict) -> Tuple[Union[List[np.ndarray], List[List[int]]], List]:
        t1 = time.time()
        self.logger.debug("Starting clustering")
        # self.experiment_metadata["clustering_hps"] = hyperparams.copy()
        # with open(hyperparams["atoms_path"], "r") as f:
        #     sem_classes = json.load(f)
        hybrid_decomp = HybridDecomp(self.analysis_pipeline, self.atoms, self.epsilons,
                                     strategy=hyperparams["strategy"], max_iterations=hyperparams["max_iterations"],
                                     min_samples=hyperparams["min_samples"], epsilon_step=hyperparams["epsilon_step"],
                                     include_outliers=hyperparams["include_outliers"]
                                     )
        layers = hybrid_decomp.cluster()
        exec_time = time.time() - t1
        self.experiment_metadata["exec_time"]["clustering"] = exec_time
        self.logger.debug("Finished clustering ({:.4f}s)".format(exec_time))
        return layers, hybrid_decomp.atoms

    def log_output(self, layers: Union[List[np.ndarray], List[List[int]]], atom_list: List):
        if self.save_output:
            t1 = time.time()
            self.logger.debug("Saving output")
            df_layers = pd.DataFrame(layers, columns=atom_list)
            df_layers["layer number"] = "layer " + df_layers.index.astype(str)
            df_layers.set_index("layer number", inplace=True)
            df_layers.to_csv(os.path.join(self.output_path, "layers.csv"))
            if self.include_metadata:
                with open(os.path.join(self.output_path, "experiment_metadata.json"), "w") as f:
                    json.dump(self.experiment_metadata, f, indent=4)
            exec_time = time.time() - t1
            self.logger.debug("Finished saving output ({:.4f}s)".format(exec_time))

    # def eval_results(self, layers: Union[List[np.ndarray], List[List[int]]], eval_handler: EvaluationHandler,
    #                  atom_list: List, log: bool = False):
    #     t1 = time.time()
    #     self.logger.debug("Evaluating results")
    #     metrics = eval_handler.metrics
    #     if log:
    #         self.logger.info("metrics: {}".format(metrics))
    #     all_results = list()
    #     for i, layer in enumerate(layers):
    #         results = eval_handler.evaluate(layer)
    #         results = {m: results[m] for m in metrics}
    #         all_results.append([self.experiment_id, i] + list(results.values()))
    #         if log:
    #             self.logger.info("{}_{}_{}".format(i, layer, results.values()))
    #     exec_time = time.time() - t1
    #     self.experiment_metadata["exec_time"]["evaluation_calc"] = exec_time
    #     self.logger.debug("Finished evaluating ({:.4f}s)".format(exec_time))
    #     if self.save:
    #         t1 = time.time()
    #         self.logger.debug("Saving results")
    #         df = pd.DataFrame(all_results, columns=["experiment_id", "layer_id"] + metrics)
    #         df.to_csv(os.path.join(self.output_path, "results.csv".format(self.experiment_id)), index=False)
    #         df_layers = pd.DataFrame(layers, columns=atom_list)
    #         df_layers.to_csv(os.path.join(self.output_path, "layers.csv"), index=False)
    #         with open(os.path.join(self.output_path, "experiment_metadata.json"), "w") as f:
    #             json.dump(self.experiment_metadata, f, indent=4)
    #         exec_time = time.time() - t1
    #         self.logger.debug("Finished saving results ({:.4f}s)".format(exec_time))

    def load_analysis(self, analysis_name: str, hyperparams: Dict, all_atoms: List[str]):
        assert analysis_name in self.data_handler.SUPPORTED_ANALYSIS
        t1 = time.time()
        self.logger.debug("Initializing {} analysis".format(analysis_name))
        # self.experiment_metadata["hyperparameters"][analysis_name] = hyperparams.copy()
        analysis_obj = self.data_handler.route_analysis(analysis_name)(hyperparams, all_atoms)
        self.analysis_pipeline.append(analysis_obj)
        self.epsilons.append(hyperparams["epsilon"])
        exec_time = time.time() - t1
        self.experiment_metadata["exec_time"][analysis_name] = exec_time
        self.logger.debug("Finished {} ({:.4f}s)".format(analysis_name, exec_time))


