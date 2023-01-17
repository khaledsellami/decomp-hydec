import os
import json

import numpy as np

from hybridDecomp import HybridDecomp
from analysis.dependencyAnalysis import DependencyAnalysis
from analysis.tfidfAnalysis import TfidfAnalysis
from analysis.sumAnalysis import SumAnalysis
from user_config import DATA_PATH


class DynamicAnalysis:
    def __init__(self, traces_path, filtered_classes=[]):
        self.traces_path = traces_path
        with open(traces_path, 'r') as file:
            traces = json.load(file)
        self.filtered_classes = filtered_classes
        self.parse_traces(traces)

    def parse_traces(self, traces):
        self.bcs = set()
        self.bcs_per_class = dict()
        self.bcs_per_DR = dict()
        self.bcs_per_IR = dict()
        for bc, bc_traces in traces.items():
            self.bcs.add(bc)
            for trace in set(bc_traces):
                trace = trace.split(",")
                for i in range(len(trace)):
                    c1 = trace[i]
                    if c1 in self.filtered_classes:
                        continue
                    if not c1 in self.bcs_per_class:
                        self.bcs_per_class[c1] = set()
                    self.bcs_per_class[c1].add(bc)
                    for j in range(i+1,len(trace)):
                        c2 = trace[j]
                        if c2 in self.filtered_classes:
                            continue
                        if j==i+1:
                            DR = (c1,c2)
                            if not DR in self.bcs_per_DR:
                                self.bcs_per_DR[DR] = set()
                            self.bcs_per_DR[DR].add(bc)
                        else:
                            IR = (c1,c2)
                            if not IR in self.bcs_per_IR:
                                self.bcs_per_IR[IR] = set()
                            self.bcs_per_IR[IR].add(bc)
        self.class_names = list(self.bcs_per_class.keys())
        self.bcs = list(self.bcs)
        self.class_relations = np.zeros((len(self.class_names),len(self.class_names)))
        self.class_bc_matrix = np.zeros((len(self.class_names),len(self.bcs)))
        for i, c1 in enumerate(self.class_names):
            for bc in self.bcs_per_class[c1]:
                self.class_bc_matrix[i,self.bcs.index(bc)] = 1
            for j, c2 in enumerate(self.class_names):
                R = (c1,c2)
                current_bcs  = set()
                if R in self.bcs_per_DR:
                    current_bcs = current_bcs.union(self.bcs_per_DR[R])
                if R in self.bcs_per_IR:
                    current_bcs = current_bcs.union(self.bcs_per_IR[R])
                self.class_relations[i,j] += len(current_bcs)


if __name__ == "__main__":
    APP_DATA_PATH = DATA_PATH
    APP = "JPetStore"
    OUTPUT_PATH = os.path.join(os.path.curdir, "logs")
    # Initialize structural analysis
    with open(os.path.join(APP_DATA_PATH, APP.lower(), "structural_data", "class_names.json"), "r") as f:
        str_classes = json.load(f)
    features = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "structural_data", "class_calls.npy"))
    structural_analysis = DependencyAnalysis(features, str_classes, str_classes)
    # Initialize semantic analysis
    with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_names.json"), "r") as f:
        sem_classes = json.load(f)
    features = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_tfidf.npy"))
    with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_words.json"), "r") as f:
        class_tokens = json.load(f)
    semantic_analysis = TfidfAnalysis(features, str_classes, str_classes, class_tokens)
    # combine structural and semantic analysis
    static_analysis = SumAnalysis([structural_analysis, semantic_analysis])
    # Initialize dynamic analysis
    dynamic_analysis_data = "/Users/khalsel/Documents/projects/hybridDecomp/applications/jpetstore/context_traces.json"
    dyn_analysis = DynamicAnalysis(dynamic_analysis_data)
    short_names = [c.split(".")[-1] for c in sem_classes]
    filtered_classes = [i for i in dyn_analysis.class_names if i.split("::")[-1] not in short_names]
    dyn_analysis = DynamicAnalysis(dynamic_analysis_data, filtered_classes)
    dynamic_analysis = DependencyAnalysis(dyn_analysis.class_relations, short_names, [i.split("::")[-1] for i in dyn_analysis.class_names])
    # Create the clustering object
    hybrid_decomp = HybridDecomp([dynamic_analysis, static_analysis], sem_classes, [0.65, 0.65])
    layers = hybrid_decomp.cluster()
    for i, layer in enumerate(layers):
        print(i, layer)

