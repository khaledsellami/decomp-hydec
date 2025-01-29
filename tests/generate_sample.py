import argparse
import os.path
import dill as pickle

import numpy as np

from hydec import generate_decomposition
from hydec.models import Partition, Granularity, Decomposition, DecompositionLayer, DecompositionLayers


def parse_layer(layer, names):
    layer = np.array(layer)
    partitions = [Partition(name="partition{}".format(p), classes=[names[i] for i in np.where(layer==p)[0]])
                  for p in np.sort(np.unique(layer))]
    return partitions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='hydec',
        description='single run of hydec or hierdec')

    parser.add_argument('APP', type=str, help='application to apply decomposition on')
    parser.add_argument("-r", "--repo", help='link for the github repository', type=str)
    parser.add_argument("-d", "--dynamic", help='path for the dynamic analysis data', type=str)
    parser.add_argument("-e", "--semantic", help='path for the semantic analysis data', type=str)
    parser.add_argument("-t", "--structural", help='path for the structural analysis data', type=str)
    parser.add_argument("-p", "--hyperparams", help='path for the hyperparameters file', type=str)
    parser.add_argument("-a", "--approach", help='use hyDec or hierDec', default="hyDec", choices=["hyDec", "hierDec"])
    parser.add_argument("-v", "--verbose", help='print the final output', action="store_true")
    args = parser.parse_args()

    app = args.APP
    app_repo = args.repo
    decomp_approach = args.approach
    hyperparams_path = args.hyperparams
    structural_path = args.structural
    semantic_path = args.semantic
    dynamic_path = args.dynamic
    verbose = args.verbose

    final_layer, layers, atoms, _ = generate_decomposition(app, app_repo, decomp_approach, hyperparams_path,
                                                             structural_path, semantic_path, dynamic_path,
                                                             include_metadata=True, save_output=False)


    partitions = [[atoms[i] for i in np.where(final_layer==p)[0]] for p in np.sort(np.unique(final_layer))]
    layers_partitions = [
        [[atoms[i] for i in np.where(layer==p)[0]] for p in np.sort(np.unique(layer))] for layer in layers]
    # decomposition = Decomposition(name=app, appName=app, language="java", level="class", appRepo=app_repo,
    #                               partitions=parse_layer(final_layer, atoms))
    # decomp_layers = [DecompositionLayer(name="layer_{}".format(i),
    #                                     decomposition=list(layers[i])) for i in range(len(layers))]
    # layer_return = DecompositionLayers(names=atoms, layers=decomp_layers, final_decomposition=decomposition)
    test_path = os.path.join(os.curdir, "tests", "tests_data", app)
    os.makedirs(test_path, exist_ok=True)
    with open(os.path.join(test_path, "decomposition.pickle"), "wb") as f:
        pickle.dump(partitions, f)
    with open(os.path.join(test_path, "layers.pickle"), "wb") as f:
        pickle.dump(layers_partitions, f)
    if verbose:
        print(layers)

