"""NumCompute-Stream: NumPy-based batch and incremental computation."""

from .benchmarking import benchmark, compare_implementations
from .io import iter_csv, load_csv
from .metrics import (
    AUC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
    StreamingAUC,
    StreamingAccuracy,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingPrecision,
    StreamingRecall,
    accuracy,
    auc,
    confusion_matrix,
    f1,
    precision,
    recall,
    roc_auc_score,
    roc_curve,
)
from .optim import grad, jacobian
from .pipeline import Compose, FeatureUnion, Pipeline
from .preprocessing import (
    ColumnSelector,
    Imputer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from .rank import StreamingRanker, StreamingTopK, percentile, rank
from .sort_search import binary_search, multi_key_sort, quickselect, stable_sort, topk
from .stats import (
    StreamingHistogram,
    StreamingStats,
    histogram,
    maximum,
    mean,
    median,
    minimum,
    percentile as stats_percentile,
    quantile,
    std,
    update_stats,
)
from .stream import StreamTrainer
from .tree import DecisionTree, DecisionTreeClassifier
from .utils import cosine_similarity, deep_sizeof, euclidean_distance, manhattan_distance, positive_int
from .visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)

__version__ = "0.2.0"

__all__ = [
    "iter_csv", "load_csv", "StandardScaler", "MinMaxScaler", "Imputer",
    "OneHotEncoder", "ColumnSelector", "stable_sort", "multi_key_sort",
    "topk", "quickselect", "binary_search", "rank", "percentile",
    "StreamingTopK", "StreamingRanker", "StreamingStats", "StreamingHistogram",
    "update_stats", "accuracy", "precision", "recall", "f1",
    "confusion_matrix", "roc_curve", "auc", "roc_auc_score", "Accuracy",
    "Precision", "Recall", "F1Score", "ConfusionMatrix", "AUC", "grad",
    "StreamingAccuracy", "StreamingPrecision", "StreamingRecall",
    "StreamingF1", "StreamingConfusionMatrix", "StreamingAUC",
    "jacobian", "Pipeline", "Compose", "FeatureUnion", "StreamTrainer",
    "DecisionTreeClassifier", "DecisionTree", "plot_metric_over_time",
    "compare_models", "plot_predictions_vs_ground_truth", "benchmark",
    "compare_implementations", "euclidean_distance", "manhattan_distance",
    "cosine_similarity", "deep_sizeof",
    "positive_int", "mean", "median", "std", "minimum", "maximum",
    "quantile", "histogram", "stats_percentile",
]
