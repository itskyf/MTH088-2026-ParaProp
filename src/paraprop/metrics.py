from typing import NamedTuple

from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)


class Metrics(NamedTuple):
    train_loss: MeanMetric
    # TODO grad_norm collection
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
    test_metrics = train_metrics.clone(prefix="val/")
    return Metrics(train_loss=MeanMetric(), train=train_metrics, test=test_metrics)
