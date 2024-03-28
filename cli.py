import argparse

from hydec import generate_decomposition


def cli(args):
    app = args.APP
    app_repo = args.repo
    decomp_approach = args.approach
    hyperparams_path = args.hyperparams
    structural_path = args.structural
    semantic_path = args.semantic
    dynamic_path = args.dynamic
    verbose = args.verbose
    granularity = args.granularity
    is_distributed = args.distributed
    decomposition, layers, atoms, _ = generate_decomposition(app, app_repo, decomp_approach, hyperparams_path,
                                                             structural_path, semantic_path, dynamic_path,
                                                             include_metadata=True, save_output=True,
                                                             granularity=granularity, is_distributed=is_distributed)
    if verbose:
        print(decomposition)


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
    parser.add_argument("-g", "--granularity", help='granularity level of the decomposition', type=str,
                        default="class", choices=["class", "method"])
    parser.add_argument("-di", "--distributed",
                        help='the application to decompose has a distributed architecture', action="store_true")
    args = parser.parse_args()
    cli(args)

