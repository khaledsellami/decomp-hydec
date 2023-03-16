import argparse
import collections
from copy import deepcopy
from multiprocessing import Pool, cpu_count

from sklearn.model_selection import ParameterGrid

from experiment import Experiment, ANALYSIS_MAP
from user_config import DATA_PATH
from utils.hyperparam_ranges import get_hyperparams

SEP = "___"


def run_experiment(args):
    params, hp_ranges, app, exp_name = args
    hyper_params = deepcopy(hp_ranges)
    for c, v in params.items():
        c = c.split(SEP)
        i, j = c[0], c[1]
        hyper_params[i][j] = v
    hps = {"clustering": hyper_params["clustering"],
           "evaluation": hyper_params["evaluation"],
           "analysis": [(i, j) for i, j in hyper_params.items() if i not in ["clustering", "evaluation"]]}
    exp = Experiment(app, hps, exp_name)
    exp.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='hydec_grid_search',
        description='Hyperparameter Grid Search for Hybrid decomposition')

    parser.add_argument('APP', type=str, help='application to apply decomposition on')
    # parser.add_argument("-s", '--seed', dest='seed', type=int, nargs='?', default=120,
    #                     help='random sampling seed')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-l", "--len", help='return the number of grid search parameters', action="store_true")
    group.add_argument("-r", "--run", help='run the decomposition once', action="store_true")
    group.add_argument("-m", "--multirun", help='run the decomposition multiple times in parallel', action="store_true")
    group.add_argument("-s", "--sequence", help='run the decomposition multiple times in sequence', action="store_true")
    parser.add_argument("-p", "--pipeline", dest='analysis_pipeline', type=str, nargs='+',
                        help='monolithic analysis processes pipeline', required=True,
                        choices=[i for i in ANALYSIS_MAP.keys() if i != "sum"])
    parser.add_argument("-j", "--jobid", type=int, help='slurm job id', required=True)
    parser.add_argument("-t", "--taskid", type=int, help='slurm task id', required=True)
    parser.add_argument("-n", '--name', dest='name', type=str, nargs='?', default="",
                        help='experiment name suffix')
    parser.add_argument("-R", "--range", type=int, help='number of processes range')
    parser.add_argument("-P", "--nprocess", type=int, help='number of processes window')
    args = parser.parse_args()
    if args.multirun and (args.range is None or args.nprocess is None):
        parser.error("--multirun requires --range and --nprocess.")
    elif args.sequence and args.range is None:
        parser.error("--sequence requires --range.")

    # seed = 120
    app = args.APP
    analysis_pipeline = args.analysis_pipeline
    name = args.name
    job_id = args.jobid
    task_id = args.taskid

    hp_ranges = get_hyperparams(["clustering", "evaluation"]+analysis_pipeline, DATA_PATH, app)
    # hp_ranges["structural"]["epsilon"] = 1
    hp_ranges["clustering"]["strategy"] = "alternating_epsilon"
    # hp_ranges["clustering"]["epsilon_step"] = 0.05
    p_grid = {"{}{}{}".format(i, SEP, j): l for i in ["clustering"]+analysis_pipeline for j, l in hp_ranges[i].items()
              if (isinstance(l, collections.Iterable) and not (isinstance(l, str) or isinstance(l, dict)))}
    p_grid = list(ParameterGrid(p_grid))
    if args.len:
        # print number of parameter and exit
        print(len(p_grid))
        print(p_grid[0].keys())
        exit()
    elif args.run:
        # run a single experiment
        print("running a single experiment on {}".format(p_grid[task_id]))
        exp_name = "{}_{}_{}".format(name, job_id, task_id)
        run_experiment((p_grid[task_id], hp_ranges, app, exp_name))
    elif args.multirun or args.sequence:
        # run multiple experiments
        exp_range = args.range
        start = task_id * exp_range
        assert start <= len(p_grid)
        end = min((task_id + 1)*exp_range, len(p_grid))
        params = p_grid[start: end]
        exp_name_format = "{}_{}_{}" if name != "" else "{}{}_{}"
        exp_names = [exp_name_format.format(name, job_id, i) for i in range(start, end)]
        if args.multirun:
            num_process = min(args.nprocess, cpu_count())
            print("running experiments from {} to {} with {} processes at a time".format(start, end, num_process))
            with Pool(processes=num_process) as p:
                p.map(run_experiment, zip(params, [hp_ranges]*len(params), [app]*len(params), exp_names))
        else:
            print("running experiments from {} to {} sequentially".format(start, end))
            for arguments in zip(params, [hp_ranges]*len(params), [app]*len(params), exp_names):
                run_experiment(arguments)
    else:
        raise NotImplementedError
