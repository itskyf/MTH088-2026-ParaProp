import torch
from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanMetric, MetricCollection

from paraprop.typing import tqdm


def evaluate(
    starting_eval_batch_size: int,
    accelerator: Accelerator,
    test_dataset: Dataset,
    model: nn.Module,
    metrics: MetricCollection,
    num_workers: int,
):
    model.eval()
    pin_memory = accelerator.device.type != "mps"

    @find_executable_batch_size(starting_batch_size=starting_eval_batch_size)
    def _execute_eval(eval_batch_size: int):
        accelerator.free_memory()
        # Ensure clean state (OOM safety)
        metrics.reset()

        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
            shuffle=False,
        )
        test_loader = accelerator.prepare_data_loader(test_loader)

        for inputs, targets in tqdm(test_loader, leave=False):
            with torch.inference_mode():
                logits = model(inputs)

            # Gather across processes while automatically removing padding
            gathered_logits, gathered_targets = accelerator.gather_for_metrics(
                (logits, targets)
            )
            metrics.update(gathered_logits, gathered_targets)

        results = metrics.compute()
        # Free memory for next stage
        metrics.reset()

        return results, eval_batch_size

    return _execute_eval()


def train_one_epoch(
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    metrics: MetricCollection,
    train_loss_metric: MeanMetric,
    max_grad_norm: float | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
    model.train()
    # Ensure clean state (OOM safety)
    metrics.reset()
    train_loss_metric.reset()

    for _, (inputs, targets) in enumerate(tqdm(dataloader, leave=False)):
        optimizer.zero_grad()

        logits = model(inputs)
        loss = loss_fn(logits, targets)
        if torch.isnan(loss) or torch.isinf(loss):
            # Divergence check
            metrics.reset()
            train_loss_metric.reset()
            return

        accelerator.backward(loss)
        if max_grad_norm and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        metrics.update(logits.detach(), targets)
        batch_size = targets.numel()
        # Update weighted loss metric to handle variable batch sizes correctly.
        train_loss_metric.update(loss.detach(), weight=batch_size)

    metrics_results = metrics.compute()
    train_loss = train_loss_metric.compute()

    # Free memory for next stage
    metrics.reset()
    train_loss_metric.reset()
    return metrics_results, train_loss
