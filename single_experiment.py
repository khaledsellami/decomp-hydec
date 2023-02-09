import os
import json

import numpy as np

from hybridDecomp import HybridDecomp
from analysis.dependencyAnalysis import DependencyAnalysis
from analysis.tfidfAnalysis import TfidfAnalysis
from analysis.sumAnalysis import SumAnalysis
from analysis.bertAnalysis import BertAnalysis
from analysis.wordEmbeddingAnalysis import WordEmbeddingAnalysis
from analysis.embeddingModel import EmbeddingModel
from user_config import DATA_PATH, DYN_DATA_PATH
from utils.logTraceParser import LogTraceParcer


if __name__ == "__main__":
    APP_DATA_PATH = DATA_PATH
    APP = "JPetStore"
    OUTPUT_PATH = os.path.join(os.path.curdir, "logs")

    # Initialize structural analysis
    # with open(os.path.join(APP_DATA_PATH, APP.lower(), "structural_data", "class_names.json"), "r") as f:
    #     str_classes = json.load(f)
    # features = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "structural_data", "class_calls.npy"))
    # structural_analysis = DependencyAnalysis(features, str_classes, str_classes)

    # # Initialize semantic analysis
    # with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_names.json"), "r") as f:
    #     sem_classes = json.load(f)
    # features = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_tfidf.npy"))
    # with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_words.json"), "r") as f:
    #     class_tokens = json.load(f)
    # semantic_analysis = TfidfAnalysis(features, str_classes, str_classes, class_tokens)

    # # combine structural and semantic analysis
    # static_analysis = SumAnalysis([structural_analysis, semantic_analysis])

    # Initialize semantic analysis with bert
    with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_names.json"), "r") as f:
        sem_classes = json.load(f)
    features = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_tfidf.npy"))
    with open(os.path.join(APP_DATA_PATH, APP.lower(), "static_analysis_results", "typeData.json"), "r") as f:
        class_text = [c["textAndNames"] for c in json.load(f)]
    embeddings = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "embeddings.npy"))
    bert_analysis = BertAnalysis(class_text, sem_classes, sem_classes, aggregation="mean", embeddings=embeddings)
    # np.save(os.path.join(APP_DATA_PATH, APP.lower(), "embeddings.npy"),bert_analysis.embeddings)

    # Initialize semantic analysis with fastext
    with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_names.json"), "r") as f:
        sem_classes = json.load(f)
    features = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_tfidf.npy"))
    with open(os.path.join(APP_DATA_PATH, APP.lower(), "static_analysis_results", "typeData.json"), "r") as f:
        class_text = [c["textAndNames"] for c in json.load(f)]
    embedding_model = EmbeddingModel()
    embedding_model.load_embedding_model()
    class_tokens = [embedding_model.tokenize(text) for text in class_text]
    emb_analysis = WordEmbeddingAnalysis(class_text, embedding_model, sem_classes, sem_classes)

    # Initialize dynamic analysis
    dynamic_analysis_data = DYN_DATA_PATH
    dyn_analysis = LogTraceParcer(dynamic_analysis_data)
    short_names = [c.split(".")[-1] for c in sem_classes]
    filtered_classes = [i for i in dyn_analysis.class_names if i.split("::")[-1] not in short_names]
    dyn_analysis = LogTraceParcer(dynamic_analysis_data, filtered_classes)
    dynamic_analysis = DependencyAnalysis(dyn_analysis.class_relations, short_names,
                                          [i.split("::")[-1] for i in dyn_analysis.class_names])

    # Create the clustering object
    hybrid_decomp = HybridDecomp([dynamic_analysis, bert_analysis], sem_classes, [0.65, 0.05])
    layers = hybrid_decomp.cluster()
    for i, layer in enumerate(layers):
        print(i, layer)

