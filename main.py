import argparse
import logging

from cli import cli
from server import serve


def main():
    # Parsing input
    parser = argparse.ArgumentParser(
        prog='msextractor',
        description='Decompose an application using HyDec or HierDec or start the Hierdec server')
    subparsers = parser.add_subparsers(dest="subtask", required=False)\
    # CLI subtask
    cli_parser = subparsers.add_parser("decompose",
                                       description="Execute a single run of hydec or hierdec")
    cli_parser.add_argument('APP', type=str, help='application to apply decomposition on')
    cli_parser.add_argument("-r", "--repo", help='link for the github repository', type=str)
    cli_parser.add_argument("-d", "--dynamic", help='path for the dynamic analysis data', type=str)
    cli_parser.add_argument("-e", "--semantic", help='path for the semantic analysis data', type=str)
    cli_parser.add_argument("-t", "--structural", help='path for the structural analysis data', type=str)
    cli_parser.add_argument("-p", "--hyperparams", help='path for the hyperparameters file', type=str)
    cli_parser.add_argument("-a", "--approach", help='use hyDec or hierDec', default="hyDec",
                            choices=["hyDec", "hierDec"])
    cli_parser.add_argument("-v", "--verbose", help='print the final output', action="store_true")
    cli_parser.add_argument("-g", "--granularity", help='granularity level of the decomposition', type=str,
                            default="class", choices=["class", "method"])
    cli_parser.add_argument("-d", "--distributed",
                            help='the application to decompose has a distributed architecture', action="store_true")
    # server subtask
    server_parser = subparsers.add_parser("start", description="start the Hierdec server")
    # configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # route the task
    args = parser.parse_args()
    if args.subtask is None or args.subtask == "start":
        serve()
    elif args.subtask == "decompose":
        cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
