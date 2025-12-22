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

        args["experiment_name"] = args.pop("name")

        simple_metrics = [
            "elapsed_time",
            "max_memory_allocated",
            "max_memory_allocated_mb",
        ]

        nested_metrics = [
            "val_metrics_min",
            "val_metrics_max",
            "test_metrics_min",
            "test_metrics_max",
            "val_metrics_mean",
            "val_metrics_std",
            "test_metrics_mean",
            "test_metrics_std",
        ]


        for metric_type in simple_metrics:
            metric_value = float(metrics[metric_type])
            metric_value = metric_value if not np.isnan(metric_value) else -1.0
            
            pulsar_metric_dict = dict(
                value=metric_value,
                name=metric_type,
                **args,
            )

            results.append(pulsar_metric_dict)


        for metric_type in nested_metrics:
            metric_type_dict = metrics[metric_type]
            for metric_type_name, metric_value in metric_type_dict.items():
                # metric_value = float(metrics[metric_name])
                metric_value = metric_value if not np.isnan(metric_value) else -1.0
                
                pulsar_metric_dict = dict(
                    value=metric_value,
                    name=f"{metric_type} {metric_type_name}",
                    **args,
                )

                results.append(pulsar_metric_dict)

        best_epoch_metric = dict(
            value=metrics["best epochs"][0],
            name="best val epoch",
            **args,
        )
        best_step_metric = dict(
            value=metrics["best steps"][0],
            name="best val step",
            **args,
        )
        results.append(best_epoch_metric)
        results.append(best_step_metric)

    except FileNotFoundError:
        pass


json_string_for_experiment = json.dumps(results)

if ndl:
    with open(ndl.json_output_file(), "w") as f_write:
        json.dump(results, f_write, indent=4)
else:
    print(json.dumps(results, indent=4))
