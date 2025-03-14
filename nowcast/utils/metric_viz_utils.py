import json
import datetime as dt
import matplotlib.pyplot as plt

from ..config import TFDataConfig


def metric_graphs(metric_dirs, offsets):
    # Load the metrics
    metrics = []
    for metric_dir in metric_dirs:
        with open(metric_dir / "metrics.json", "r") as f:
            metric = json.load(f)
            metrics.append(metric)
            if metrics:
                assert set(metrics[0].keys()) == set(
                    metric.keys()
                ), "Models have different metrics trained."

    output_dir = (
        TFDataConfig.TB_LOG_DIR
        / f"metric_viz_{dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir.mkdir()

    with open(output_dir / "metadata.txt", "w") as f:
        f.write("Models used for visualization:\n")
        for metric_dir in metric_dirs:
            f.write(f"{metric_dir.parent}\n")

    for metric in metrics[0].keys():
        x = []
        y = []
        for all_metric, offset in zip(metrics, offsets):
            x += list(
                range(
                    offset + 1,
                    offset + TFDataConfig.HEM_WINDOW_SIZE + 1,
                )
            )
            y += all_metric[metric]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, marker="o")
        ax.set_title(f"{metric.upper()}")

        x_labels = [f"{int(x_val * 30)} mins" for x_val in x]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)

        ax.set_xlabel("Forecast Time")
        ax.set_ylabel(f"{metric}")
        ax.grid(True)
        fig.tight_layout()

        fig.savefig(output_dir / f"{metric}.png")
        plt.close(fig)
