from pathlib import Path
from typing import List, Dict, Tuple
import sys
from ast import literal_eval
from itertools import product
import argparse

sys.path.append("../")
sys.path.append("./")

from run_single_experiment import get_args

_default_params, parser = get_args(add_name=False)


try:
    import nirvana_dl as ndl
    PARAMS = ndl.params()
    print("Imported Nirvana DL package", file=sys.stderr)
except ImportError:
    print("Couldn't import `nirvana_dl` package", file=sys.stderr)
    ndl = None
    PARAMS = vars(_default_params)
    
    for key in PARAMS:
        PARAMS[key] = str(PARAMS[key])
    PARAMS["name"] = ["Name", "ASDJAHSGDKHJAGSDJKHGASJKD"]



store_true_args = []
for action in parser._actions:
    if isinstance(action, argparse._StoreTrueAction):
        store_true_args.append(action.option_strings[0][2:])


STORE_TRUE_ARGS = set(store_true_args)


def filter_params_on_single_and_multiple_options():
    params_with_multiple_options_and_values: Dict[str, List[str]] = {}  # those which can be casted to lists of values
    params_with_single_options_and_values: Dict[str, str] = {}  # those which can be casted to lists of values

    def _check_if_iterable(value):
        try:
            s = literal_eval(repr(value))
            if isinstance(s, (list, tuple, set)):
                return True
            return False
        except (ValueError, SyntaxError):
            return isinstance(value, (list, tuple, set))

    for param, value in PARAMS.items():
        if _check_if_iterable(value=value):
            params_with_multiple_options_and_values[param] = [str(x) for x in literal_eval(repr(value))]
        else:
            params_with_single_options_and_values[param] = value

    return params_with_single_options_and_values, params_with_multiple_options_and_values


def create_one_run(params_flattened_one_instance: Dict[str, str]):
    launch_script_string_container: List[List[str]] = ["python", "run_single_experiment.py"]

    for option_name, option_value in params_flattened_one_instance.items():
        print(f"Option: {repr(option_name)}, param: {repr(option_value)}", file=sys.stderr)
        if option_name == "time_based_features_types" and option_value == "no_encodings":  # it signals not to use time-based encodings, we need to explicitly specify it as an empty argument
            option_value = "  "

        if option_value == "None" or option_value is None:
            continue

        if option_name in STORE_TRUE_ARGS:  # this option value is true and it;s passed as true
            if option_value == "True" or option_value == True:  # it must remain with this explicit option as we want to distinguish it from other values!
                param_string: str = f"--{option_name}"
            else:
                continue
        else:
            param_string = f"--{option_name} {option_value}"

        launch_script_string_container.append(param_string)

    launch_script_string: str = " ".join(launch_script_string_container)

    return launch_script_string


if __name__ == "__main__":
    single_choice_params, multi_choice_params = filter_params_on_single_and_multiple_options()
    print("#!/bin/bash")

    if len(multi_choice_params) > 0:
        multi_choice_containers_per_param: List[List[Tuple[str, str]]] = []

        for param, values in multi_choice_params.items():
            multi_choice_containers_per_param.append([(param, v) for v in values])

        print(f"{multi_choice_containers_per_param=}", file=sys.stderr)
        multi_choice_params_product = product(*multi_choice_containers_per_param)

        launch_strings: List[str] = []

        for params_choice in multi_choice_params_product:
            print(f"{params_choice}", file=sys.stderr)
            parameters = {p: v for p, v in params_choice}
            parameters.update(single_choice_params)

            one_run_string = create_one_run(parameters)

            launch_strings.append(one_run_string)

        print("\n\n".join(launch_strings))
    else:
        print(create_one_run(single_choice_params))
