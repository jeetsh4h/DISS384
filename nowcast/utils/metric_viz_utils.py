import json
import datetime as dt
import matplotlib.pyplot as plt

from ..config import TFDataConfig


def metric_graphs(metric_dirs, offsets, flow=False):
    # Load the metrics
    metrics = []
    for metric_dir in metric_dirs:
        metric_file = metric_dir = metric_dir / (
            "metrics.json" if not flow else "flow_metrics.json"
        )
        with open(metric_file, "r") as f:
            metric = json.load(f)
            metrics.append(metric)
            if metrics:
                assert set(metrics[0].keys()) == set(
                    metric.keys()
                ), "Models have different metrics trained."

    metric_viz_dir_name = f"metric_viz_{dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime('%Y%m%d-%H%M%S')}"
    output_dir = TFDataConfig.TB_LOG_DIR / (
        metric_viz_dir_name if not flow else f"{metric_viz_dir_name}_flow"
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
        ax.plot(x, y, ("b-o" if not flow else "r-s"))
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


def overlay_metric_graphs(metric_dirs, flow_metric_dirs, offsets):
    """Visualize model metrics and flow metrics overlaid in the same graphs."""
    # Load the metrics
    model_metrics = []
    flow_metrics = []

    # Load model metrics
    for metric_dir in metric_dirs:
        metric_file = metric_dir / "metrics.json"
        with open(metric_file, "r") as f:
            metric = json.load(f)
            model_metrics.append(metric)
            if model_metrics:
                assert set(model_metrics[0].keys()) == set(
                    metric.keys()
                ), "Models have different metrics trained."

    # Load flow metrics
    for metric_dir in flow_metric_dirs:
        metric_file = metric_dir / "flow_metrics.json"
        with open(metric_file, "r") as f:
            metric = json.load(f)
            flow_metrics.append(metric)
            if flow_metrics:
                assert set(flow_metrics[0].keys()) == set(
                    metric.keys()
                ), "Flow models have different metrics trained."

    # Ensure both model and flow metrics have the same keys
    shared_metrics = set(model_metrics[0].keys()).intersection(
        set(flow_metrics[0].keys())
    )

    metric_viz_dir_name = f"metric_viz_overlay_{dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime('%Y%m%d-%H%M%S')}"
    output_dir = TFDataConfig.TB_LOG_DIR / metric_viz_dir_name
    output_dir.mkdir()

    with open(output_dir / "metadata.txt", "w") as f:
        f.write("Models used for visualization:\n")
        for metric_dir in metric_dirs:
            f.write(f"{metric_dir.parent} (model)\n")
        for metric_dir in flow_metric_dirs:
            f.write(f"{metric_dir.parent} (flow)\n")

    for metric in shared_metrics:
        model_x = []
        model_y = []
        flow_x = []
        flow_y = []

        # Process model metrics
        for all_metric, offset in zip(model_metrics, offsets):
            model_x += list(
                range(
                    offset + 1,
                    offset + TFDataConfig.HEM_WINDOW_SIZE + 1,
                )
            )
            model_y += all_metric[metric]

        # Process flow metrics
        for all_metric, offset in zip(flow_metrics, offsets):
            flow_x += list(
                range(
                    offset + 1,
                    offset + TFDataConfig.HEM_WINDOW_SIZE + 1,
                )
            )
            flow_y += all_metric[metric]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(model_x, model_y, "b-o", label="Model")
        ax.plot(flow_x, flow_y, "r-s", label="Flow")
        ax.set_title(f"{metric.upper()}")

        x_labels = [f"{int(x_val * 30)} mins" for x_val in model_x]
        ax.set_xticks(model_x)
        ax.set_xticklabels(x_labels, rotation=45)

        ax.set_xlabel("Forecast Time")
        ax.set_ylabel(f"{metric}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        fig.savefig(output_dir / f"{metric}.png")
        plt.close(fig)


def training_graphs(history, offset):
    loss = history["loss"]
    val_loss = history["val_loss"]

    assert len(loss) == len(val_loss)

    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss, "b-o", label="Training loss")
    ax.plot(epochs, val_loss, "r-s", label="Validation loss")
    ax.set_title(f"Loss (nowcast offset of {(offset * 0.5):.0f} hours)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(epochs)
    fig.tight_layout()

    return fig
