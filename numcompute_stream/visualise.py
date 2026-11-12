"""Reusable Matplotlib plots for streaming metrics and predictions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _finish(fig: Any, save_path: str | None, show: bool) -> None:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()


def plot_metric_over_time(metric_values: Any, title: str = "Metric over time", ylabel: str = "Metric", *, save_path: str | None = None, show: bool = False):
    values = np.asarray(metric_values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"metric_values must be 1D; received shape {values.shape}.")
    fig, ax = plt.subplots()
    ax.plot(np.arange(values.size), values, marker="o")
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path, show)
    return fig, ax


def compare_models(metric1: Any, metric2: Any, labels: Sequence[str], *, title: str = "Model comparison", ylabel: str = "Metric", save_path: str | None = None, show: bool = False):
    first, second = np.asarray(metric1, dtype=float), np.asarray(metric2, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("model metric histories must be equal-length 1D arrays.")
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two model names.")
    fig, ax = plt.subplots()
    x = np.arange(first.size)
    ax.plot(x, first, marker="o", label=labels[0])
    ax.plot(x, second, marker="s", label=labels[1])
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path, show)
    return fig, ax


def plot_predictions_vs_ground_truth(y_true: Any, y_pred: Any, *, title: str = "Predictions vs ground truth", save_path: str | None = None, show: bool = False):
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must be equal-length 1D arrays; received {true.shape} and {pred.shape}.")
    fig, ax = plt.subplots()
    x = np.arange(true.size)
    ax.plot(x, true, "o-", label="Ground truth")
    ax.plot(x, pred, "x--", label="Predictions")
    ax.set(title=title, xlabel="Sample", ylabel="Class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path, show)
    return fig, ax


__all__ = ["plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth"]
