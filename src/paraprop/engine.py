import contextlib

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


def train_one_epoch_minibatch(
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    grad_norm_metrics: MetricCollection,
    metrics: MetricCollection,
    train_loss_metric: MeanMetric,
    max_grad_norm: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
    """Trains the model for one epoch using mini-batch gradient updates.

    This function performs a forward pass, backward pass, and optimizer step
    for every batch in the dataloader. It represents modern deep learning
    practices where weights are updated frequently within a single epoch.

    Args:
        accelerator (Accelerator): Hugging Face Accelerator for distributed training.
        dataloader (DataLoader): Iterable over the training dataset.
        model (nn.Module): The neural network model to be trained.
        loss_fn (nn.Module): The objective function to calculate the loss.
        optimizer (optim.Optimizer): The optimization algorithm.
        grad_norm_metrics (MetricCollection): Metrics to track gradient norms.
        metrics (MetricCollection): Task-specific metrics (e.g., accuracy).
        train_loss_metric (MeanMetric): Metric to track the average training loss.
        max_grad_norm (float): Maximum allowed gradient norm for clipping.

    Returns:
        tuple[dict[str, torch.Tensor], torch.Tensor] | None: A tuple containing
        grad_norm_results, metrics_results, and train_loss. Returns None if
        divergence (NaN/Inf loss) is detected.
    """
    model.train()
    # Ensure clean state (OOM safety)
    grad_norm_metrics.reset()
    metrics.reset()
    train_loss_metric.reset()

    for _, (inputs, targets) in enumerate(tqdm(dataloader, leave=False)):
        optimizer.zero_grad()

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        # Divergence check - at least 1 GPU has NaN/Inf
        is_bad_local = (~torch.isfinite(loss.detach())).to(torch.int)
        is_bad_global = accelerator.reduce(is_bad_local, reduction="sum")
        if is_bad_global.item() > 0:
            grad_norm_metrics.reset()
            metrics.reset()
            train_loss_metric.reset()
            return None

        accelerator.backward(loss)

        # Standard optimizer: step every batch
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norm_metrics.update(grad_norm)

        optimizer.step()

        # Update weighted loss metric to handle variable batch sizes correctly.
        metrics.update(logits.detach(), targets)
        train_loss_metric.update(loss.detach(), weight=targets.numel())

    grad_norm_results = grad_norm_metrics.compute()
    metrics_results = metrics.compute()
    train_loss = train_loss_metric.compute()

    # Free memory for next stage
    grad_norm_metrics.reset()
    metrics.reset()
    train_loss_metric.reset()

    return grad_norm_results, metrics_results, train_loss


def train_one_epoch_fullbatch(
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    grad_norm_metrics: MetricCollection,
    metrics: MetricCollection,
    train_loss_metric: MeanMetric,
    max_grad_norm: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
    """Trains the model for one epoch using full-batch (epoch-level) updates.

    This function accumulates gradients across all mini-batches in the epoch
    and performs a single optimizer step at the very end. This matches the
    original evaluation regime used in early methods like Fahlman's QuickProp.

    Args:
        accelerator (Accelerator): Hugging Face Accelerator for distributed training.
        dataloader (DataLoader): Iterable over the training dataset.
        model (nn.Module): The neural network model to be trained.
        loss_fn (nn.Module): The objective function to calculate the loss.
        optimizer (optim.Optimizer): The optimization algorithm.
        grad_norm_metrics (MetricCollection): Metrics to track gradient norms.
        metrics (MetricCollection): Task-specific metrics (e.g., accuracy).
        train_loss_metric (MeanMetric): Metric to track the average training loss.
        max_grad_norm (float): Maximum allowed gradient norm for clipping.

    Returns:
        tuple[dict[str, torch.Tensor], torch.Tensor] | None: A tuple containing
        grad_norm_results, metrics_results, and train_loss. Returns None if
        divergence (NaN/Inf loss) is detected.
    """
    model.train()
    # Ensure clean state (OOM safety)
    grad_norm_metrics.reset()
    metrics.reset()
    train_loss_metric.reset()

    num_steps = len(dataloader)

    # Accumulate grads across the whole epoch
    optimizer.zero_grad()

    for step_idx, (inputs, targets) in enumerate(tqdm(dataloader, leave=False)):
        # Disable inter-GPU sync until accumulation finish
        sync_ctx = (
            accelerator.no_sync(model)
            if (accelerator.num_processes > 1 and step_idx < (num_steps - 1))
            else contextlib.nullcontext()
        )

        with sync_ctx:
            logits = model(inputs)
            loss = loss_fn(logits, targets)

            # Divergence check - at least 1 GPU has NaN/Inf
            is_bad_local = (~torch.isfinite(loss.detach())).to(torch.int)
            is_bad_global = accelerator.reduce(is_bad_local, reduction="sum")
            if is_bad_global.item() > 0:
                grad_norm_metrics.reset()
                metrics.reset()
                train_loss_metric.reset()
                return None

            # Scale loss so the accumulated gradient is the mean over the entire epoch
            loss_for_backward = loss * (targets.numel() / len(dataloader.dataset))
            accelerator.backward(loss_for_backward)

        # Update metrics per batch (does not affect gradients)
        metrics.update(logits.detach(), targets)
        train_loss_metric.update(loss.detach(), weight=targets.numel())

    # Step ONCE at the end of the epoch
    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
    grad_norm_metrics.update(grad_norm)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    grad_norm_results = grad_norm_metrics.compute()
    metrics_results = metrics.compute()
    train_loss = train_loss_metric.compute()

    # Free memory for next stage
    grad_norm_metrics.reset()
    metrics.reset()
    train_loss_metric.reset()

    return grad_norm_results, metrics_results, train_loss
