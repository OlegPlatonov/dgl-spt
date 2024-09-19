import yaml
import json
from pathlib import Path
import numpy as np

try:
    import nirvana_dl as ndl
except ImportError:
    ndl = None
    
    
experimetal_results_dir: Path = Path(__file__).parent.parent / "experiments"


results = []
for exp_dir in experimetal_results_dir.glob("*/*"):
    try:
        print(exp_dir)    

        args_file = exp_dir / "args.yaml" 
        metrics_file = exp_dir / "metrics.yaml"


        args = yaml.safe_load(open(args_file))
        metrics = yaml.safe_load(open(metrics_file))

        print(f"{args_file=}")
        print(f"{metrics_file=}")

        TAG = args["dataset"].replace("-", "_")

        metric_name = args["metric"]
        
        args.pop("name")


        metric_field_to_pulsar_unified = {
            f'val {metric_name} mean': f"val_metric_mean_{TAG}",
            f'val {metric_name} std': f"val_metric_std_{TAG}",
            f'test {metric_name} mean': f"test_metric_mean_{TAG}",
            f'test {metric_name} std': f"test_metric_std_{TAG}",
        }

        for metric_in_script, metric_for_pulsar_corresponding_name in sorted(metric_field_to_pulsar_unified.items()):
            metric_value = metrics[metric_in_script]
            metric_value = metric_value if not np.isnan(metric_value) else -1.0
            
            pulsar_metric_dict = dict(
                value=metric_value,
                name=metric_for_pulsar_corresponding_name,
                **args,
            )

            results.append(pulsar_metric_dict)
    except FileNotFoundError:
        pass


json_string_for_experiment = json.dumps(results)

if ndl:
    with open(ndl.json_output_file(), "w") as f_write:
        json.dump(results, f_write, indent=4)
else:
    print(results)