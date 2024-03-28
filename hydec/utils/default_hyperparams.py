from typing import List, Dict, Union


def generate_hps_input(pipeline: List[Union[str, List[str]]], hp_ranges) -> Dict[str, Dict]:
    hps = dict()
    for k in pipeline:
        if isinstance(k, list):
            hps["sum"] = hp_ranges["sum"].copy()
            hps["sum"]["analysis"] = generate_hps_input(k, hp_ranges)
        else:
            hps[k] = hp_ranges[k]
    return hps


def merge_input(inputs: Dict[str, Dict], defaults: Dict[str, Dict]) -> Dict[str, Dict]:
        output = dict()
        for c in set(inputs.keys()).union(set(defaults.keys())):
            if c in inputs and c in defaults:
                output[c] = dict()
                for k, v in defaults[c].items():
                    output[c][k] = v
                for k, v in inputs[c].items():
                    output[c][k] = v
            elif c in inputs:
                output[c] = inputs[c]
            else:
                output[c] = defaults[c]
        return output


def get_default_hyperparams() -> Dict[str, Dict]:

    hp_ranges = dict()

    hp_ranges["dynamic"] = dict(
       epsilon=0.65,
       similarity="call"
   )

    hp_ranges["structural"] = dict(
       epsilon=0.65,
       similarity="call"
   )

    hp_ranges["semantic"] = dict(
        epsilon=0.65,
        aggregation="mean"
    )

    hp_ranges["sum"] = dict(
        analysis=None,
        weights=None,
        epsilon=0.65
    )

    hp_ranges["clustering"] = dict(
        strategy="alternating_epsilon",
        max_iterations=1000,
        min_samples=2,
        epsilon_step=0.05,
        include_outliers=False
    )
    return hp_ranges
    # return generate_hps_input(names, hp_ranges)
