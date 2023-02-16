import os
import json
import datetime
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from hybridDecomp import HybridDecomp
from analysis.dependencyAnalysis import DependencyAnalysis
from analysis.tfidfAnalysis import TfidfAnalysis
from analysis.sumAnalysis import SumAnalysis
from analysis.bertAnalysis import BertAnalysis
from analysis.wordEmbeddingAnalysis import WordEmbeddingAnalysis
from analysis.embeddingModel import EmbeddingModel
from user_config import DATA_PATH, DYN_DATA_PATH
from utils.logTraceParser import LogTraceParcer
from evaluation.evaluationHandler import EvaluationHandler


if __name__ == "__main__":
    APP_DATA_PATH = DATA_PATH
    APP = "JPetStore"
    start_date = datetime.datetime.now()
    experiment_id = "hyDec_{}_{}".format(APP, start_date.strftime("%Y%m%d%H%M%S"))
    output_path = os.path.join(os.path.curdir, "logs", experiment_id)
    experiment_metadata = dict()
    hyperparams = dict()
    analysis_pipeline = list()
    epsilons = list()

    experiment_metadata["experiment_id"] = experiment_id
    experiment_metadata["application"] = APP
    experiment_metadata["start_datetime"] = start_date.strftime("%Y-%m-%d %H:%M:%S")
    experiment_metadata["hyperparameters"] = hyperparams

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    start_time = time.time()

    # Initialize structural analysis
    # print("Initializing structural analysis")
    # with open(os.path.join(APP_DATA_PATH, APP.lower(), "structural_data", "class_names.json"), "r") as f:
    #     str_classes = json.load(f)
    # features = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "structural_data", "class_calls.npy"))
    # structural_analysis = DependencyAnalysis(features, str_classes, str_classes)

    # # Initialize semantic analysis
    # print("Initializing semantic analysis")
    # with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_names.json"), "r") as f:
    #     sem_classes = json.load(f)
    # features = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_tfidf.npy"))
    # with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_words.json"), "r") as f:
    #     class_tokens = json.load(f)
    # semantic_analysis = TfidfAnalysis(features, str_classes, str_classes, class_tokens)

    # # combine structural and semantic analysis
    # print("Combining structural and semantic analysis")
    # static_analysis = SumAnalysis([structural_analysis, semantic_analysis])

    # Initialize semantic analysis with bert
    # print("Initializing semantic analysis with bert")
    # with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_names.json"), "r") as f:
    #     sem_classes = json.load(f)
    # features = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_tfidf.npy"))
    # with open(os.path.join(APP_DATA_PATH, APP.lower(), "static_analysis_results", "typeData.json"), "r") as f:
    #     class_text = [c["textAndNames"] for c in json.load(f)]
    # embeddings = np.load(os.path.join(APP_DATA_PATH, APP.lower(), "embeddings.npy"))
    # bert_analysis = BertAnalysis(class_text, sem_classes, sem_classes, aggregation="mean", embeddings=embeddings)
    # np.save(os.path.join(APP_DATA_PATH, APP.lower(), "embeddings.npy"),bert_analysis.embeddings)

    # Initialize semantic analysis with fastext
    print("Initializing semantic analysis with fastext")
    t1 = time.time()
    with open(os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_names.json"), "r") as f:
        sem_classes = json.load(f)
    with open(os.path.join(APP_DATA_PATH, APP.lower(), "static_analysis_results", "typeData.json"), "r") as f:
        class_text = [c["textAndNames"] for c in json.load(f)]
    embedding_model = EmbeddingModel()
    embedding_model.load_embedding_model()
    class_tokens = [embedding_model.tokenize(text) for text in class_text]
    emb_analysis = WordEmbeddingAnalysis(class_text, embedding_model, sem_classes, sem_classes)
    analysis_pipeline.append(emb_analysis)
    epsilons.append(0.15)
    hyperparams["word_embedding_hps"] = dict()
    hyperparams["word_embedding_hps"]["features_path"] = os.path.join(
        APP_DATA_PATH, APP.lower(), "static_analysis_results", "typeData.json"
    )
    hyperparams["word_embedding_hps"]["aggregation"] = "mean"
    hyperparams["word_embedding_hps"]["model_type"] = "fasttext"
    hyperparams["word_embedding_hps"]["model_pooling_approach"] = "avg"
    hyperparams["word_embedding_hps"]["model_dim"] = 300
    hyperparams["word_embedding_hps"]["fine_tuned"] = False
    print("Finished semantic analysis with fastext ({:.4f}s)".format(time.time()-t1))

    # Initialize dynamic analysis
    print("Initializing dynamic analysis")
    t1 = time.time()
    dynamic_analysis_data = DYN_DATA_PATH
    dyn_analysis = LogTraceParcer(dynamic_analysis_data)
    short_names = [c.split(".")[-1] for c in sem_classes]
    filtered_classes = [i for i in dyn_analysis.class_names if i.split("::")[-1] not in short_names]
    dyn_analysis = LogTraceParcer(dynamic_analysis_data, filtered_classes)
    dynamic_analysis = DependencyAnalysis(dyn_analysis.class_relations, short_names,
                                          [i.split("::")[-1] for i in dyn_analysis.class_names])
    analysis_pipeline.append(dynamic_analysis)
    epsilons.append(0.65)
    hyperparams["dynamic_hps"] = dict()
    hyperparams["dynamic_hps"]["features_path"] = dynamic_analysis_data
    hyperparams["dynamic_hps"]["similarity"] = "call"
    print("Finished dynamic analysis ({:.4f}s)".format(time.time()-t1))

    # Initialize evaluation class
    print("Initializing evaluation class")
    t1 = time.time()
    threshold = 50
    structural_data_path = os.path.join(APP_DATA_PATH, APP.lower(), "structural_data", "interactions.npy")
    #semantic_data_path = os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_tfidf.npy")
    semantic_data_path = os.path.join(APP_DATA_PATH, APP.lower(), "semantic_data", "class_words.json")
    with open(semantic_data_path, "r") as f:
        class_words = json.load(f)
    semantic_data = CountVectorizer().fit_transform([" ".join(i) for i in class_words]).toarray()
    semantic_data = semantic_data[:, semantic_data.sum(axis=0) < threshold]
    semantic_data = semantic_data.dot(semantic_data.transpose())
    eval_handler = EvaluationHandler(structural_data_path, semantic_data)
    eval_hyperparams = dict()
    experiment_metadata["evaluation_hps"] = eval_hyperparams
    metrics = eval_handler.DEFAULT_METRICS
    eval_hyperparams["metrics"] = metrics
    eval_hyperparams["structural_data_path"] = structural_data_path
    eval_hyperparams["semantic_data_path"] = semantic_data_path
    eval_hyperparams["semantic_data_threshold"] = threshold
    eval_hyperparams["ned_method"] = "minmax"
    eval_hyperparams["ned_ranges"] = (5, 19)
    print("Finished initializing evaluation class ({:.4f}s)".format(time.time()-t1))

    # Create the clustering object
    print("Starting the clustering")
    t1 = time.time()
    strategy = "alternating_epsilon"
    hybrid_decomp = HybridDecomp(analysis_pipeline, sem_classes, epsilons, strategy=strategy)
    layers = hybrid_decomp.cluster()
    hyperparams["analysis_pipeline"] = [str(i.__class__.__name__) for i in analysis_pipeline]
    hyperparams["epsilons"] = epsilons
    hyperparams["clustering_hps"] = dict()
    hyperparams["clustering_hps"]["max_iterations"] = 50
    hyperparams["clustering_hps"]["min_sample"] = 2
    hyperparams["clustering_hps"]["strategy"] = strategy
    hyperparams["clustering_hps"]["epsilon_step"] = 0.05
    hyperparams["clustering_hps"]["include_outliers"] = False
    print("Finished the clustering ({:.4f}s)".format(time.time()-t1))

    # recording execution time
    experiment_metadata["execution time"] = start_time - time.time()

    # Show the results
    print("Showing and saving the results")
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(experiment_metadata, f)
    all_results = list()
    for i, layer in enumerate(layers):
        results = eval_handler.evaluate(layer)
        all_results.append([experiment_id, i] + list(results.values()))
        print(i, layer, results.values())
    print(results.keys())
    df = pd.DataFrame(all_results,
                      columns=["experiment_id", "layer_id"]+metrics)
    df.to_csv(os.path.join(output_path, "results.csv".format(experiment_id)), index=False)
    df_layers = pd.DataFrame(layers, columns=sem_classes)
    df_layers.to_csv(os.path.join(output_path, "layers.csv"), index=False)
    # with open(os.path.join(output_path, "layers.json"), "w") as f:
    #     json.dump(layers, f)


