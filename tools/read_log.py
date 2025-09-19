import csv
import sys


def read_file(path: str) -> dict:
    with open(path, "r") as f:
        lines = f.readlines()[-1000:]
    metrics = {}
    for line in lines:
        if "wandb:" not in line:
            continue
        try:
            k, v = line.split()[-2:]
            dataset = k.split("/")[1]
        except Exception:
            continue
        if k.endswith("acc/mean@8"):
            metric = "Acc-Pass@1"
        elif k.endswith("acc/best@8/mean"):
            metric = "Acc-Pass@8"
        elif k.endswith("len/mean@8"):
            metric = "Len-Pass@1"
        elif k.endswith("len/best@8/mean"):
            metric = "Len-Pass@8"
        else:
            continue
        try:
            value = float(v) * (100 if metric.startswith("Acc") else 1)
            metrics[f"{dataset}-{metric}"] = value
        except Exception:
            continue

    return metrics


if __name__ == "__main__":
    data = {}
    for file in sys.argv[1:]:
        data[file.rsplit(".", 1)[0]] = read_file(file)
    # Collect all unique metric keys across files to form columns
    all_metrics = set()
    for metrics in data.values():
        all_metrics.update(metrics.keys())
    columns = ["filename"] + sorted(all_metrics)

    # Write CSV manually
    with open("metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for filename, metrics in sorted(data.items()):
            row = [filename] + [metrics.get(col, "") for col in columns[1:]]
            writer.writerow(row)
