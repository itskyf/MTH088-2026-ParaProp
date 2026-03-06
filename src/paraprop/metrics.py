from typing import NamedTuple

from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class Metrics(NamedTuple):
    grad_norm: MetricCollection
    train_loss: MeanMetric
    train: MetricCollection
    test: MetricCollection


def build_metrics(num_classes: int) -> Metrics:
    train_metrics = MetricCollection(
        MetricCollection(
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassF1Score(num_classes=num_classes, average="macro"),
        ),
        prefix="train/",
    )
    grad_norm = MetricCollection([MeanMetric(), MaxMetric()], prefix="train/grad_norm/")
    test_metrics = train_metrics.clone(prefix="val/")
    return Metrics(
        grad_norm=grad_norm,
        train_loss=MeanMetric(),
        train=train_metrics,
        test=test_metrics,
    )
