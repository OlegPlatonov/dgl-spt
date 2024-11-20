import json
import sys

sys.path.append("../")
sys.path.append("./")

from run_single_experiment import get_args

arguments, _ = get_args(add_name=False)



arguments_dict = vars(arguments)
print(f"{arguments_dict=}")




with open("config.txt", "w") as f_write:
    print('{', file=f_write)
    
    parameter_string = "    \"name\": ${{global.name!\"Name1\"}},"
    print(parameter_string, file=f_write)
    for argument, default_value in arguments_dict.items():
        parameter_string = f"    \"{argument}\": ${{global.{argument}!\"{default_value}\"}},"
        print(parameter_string, file=f_write)
    
    print('}', file=f_write)

json.dump(arguments_dict, fp=open("json_new.json", "w"))
