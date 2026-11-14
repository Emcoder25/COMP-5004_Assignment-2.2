from __future__ import annotations

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency for plotting
    plt = None
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None


def _require_matplotlib():
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. "
            f"Original error: {_MATPLOTLIB_ERROR}"
        )


def plot_metric_over_time(
    metric_values,
    title: str,
    ylabel: str,
    save_path: str | None = None,
    show: bool = False,
):
    _require_matplotlib()
    values = np.asarray(metric_values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(values.size), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax


def compare_models(
    metric1,
    metric2,
    labels,
    title: str = "Model comparison",
    ylabel: str = "Metric",
    save_path: str | None = None,
    show: bool = False,
):
    _require_matplotlib()
    values1 = np.asarray(metric1, dtype=float)
    values2 = np.asarray(metric2, dtype=float)
    labels = list(labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.plot(x, values1, marker="o", label=labels[0])
    ax.plot(x, values2, marker="s", label=labels[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title: str = "Predictions vs ground truth",
    save_path: str | None = None,
    show: bool = False,
):
    _require_matplotlib()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(y_true))
    ax.plot(x, y_true, marker="o", label="Actual")
    ax.plot(x, y_pred, marker="x", label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Label")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax


__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
]
