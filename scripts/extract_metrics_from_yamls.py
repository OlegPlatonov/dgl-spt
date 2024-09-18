import yaml
import json
from pathlib import Path
import numpy as np

try:
    import nirvana_dl as ndl
except ImportError:
    ndl = None
    
    
experimetal_results_dir: Path = Path(__file__).parent.parent / "experiments"


args_file = next(experimetal_results_dir.glob("*/*/args.yaml"))
metrics_file = next(experimetal_results_dir.glob("*/*/metrics.yaml"))


args = yaml.safe_load(open(args_file))
metrics = yaml.safe_load(open(metrics_file))

# print(f"{args_file=}")
# print(f"{metrics_file=}")

TAG = args["dataset"].replace("-", "_")

metric_name = args["metric"]


metric_field_to_pulsar_unified = {
    f'val {metric_name} mean': f"val_metric_mean_{TAG}",
    f'val {metric_name} std': f"val_metric_std_{TAG}",
    f'test {metric_name} mean': f"test_metric_mean_{TAG}",
    f'test {metric_name} std': f"test_metric_std_{TAG}",
}


metrics_for_pulsar = []

for metric_in_script, metric_for_pulsar_corresponding_name in sorted(metric_field_to_pulsar_unified.items()):
    metric_value = metrics[metric_in_script]
    metric_value = metric_value if not np.isnan(metric_value) else -1.0
    
    is_main = TAG == "traffic_jams" and metric_for_pulsar_corresponding_name.startswith("test_metric_mean")
    pulsar_metric_dict = dict(
        value=metric_value,
        name=metric_for_pulsar_corresponding_name,
        tags=[TAG],
        main=is_main,
    )

    metrics_for_pulsar.append(pulsar_metric_dict)

json_string = json.dumps(metrics_for_pulsar)



if ndl:
    with open(ndl.json_output_file(), "w") as f_write:
        json.dump(metrics_for_pulsar, f_write)
else:
    print(json_string)