from pathlib import Path

from accelerate import Accelerator
from accelerate.tracking import TrackioTracker
from accelerate.utils import (
    DynamoBackend,
    LoggerType,
    ProjectConfiguration,
    TorchDynamoPlugin,
    find_executable_batch_size,
    set_seed,
)
from hydra.core.hydra_config import HydraConfig
from hydra_zen import store, zen
from hydra_zen.typing import Partial
from torch import nn, optim
from torch.utils.data import DataLoader

import paraprop
from paraprop.datasets import build_datasets
from paraprop.engine import evaluate, train_one_epoch
from paraprop.metrics import build_metrics
from paraprop.models import ParaConv
from paraprop.typing import tqdm


def get_run_name(
    optimizer_fn: Partial[optim.Optimizer],
    max_grad_norm: float | None,
) -> str:
    optimizer_name = f"optimizer:{optimizer_fn.func.__name__}"
    grad_clip_str = f"grad_clip:{max_grad_norm or 'off'}"
    return "-".join([optimizer_name, grad_clip_str])


def train_and_eval(
    # Experiment setup
    optimizer_fn: Partial[optim.Optimizer],
    max_grad_norm: float | None,
    num_epochs: int,
    seed: int,
    # CLI positional
    root: Path,
    # CLI optional
    checkpoint_total_limit: int = 5,
    initial_train_batch_size: int = 1024,
    num_workers: int = 0,
):
    set_seed(seed)

    accelerator = Accelerator(
        dynamo_plugin=TorchDynamoPlugin(
            backend=DynamoBackend.INDUCTOR, mode="max-autotune"
        ),
        log_with=[LoggerType.TRACKIO],
        project_config=ProjectConfiguration(
            project_dir=HydraConfig.get().runtime.output_dir,
            automatic_checkpoint_naming=True,
            total_limit=checkpoint_total_limit,
        ),
    )
    accelerator.init_trackers(
        project_name=paraprop.__name__,
        config={"seed": seed, "max_grad_norm": max_grad_norm, "num_epochs": num_epochs},
        init_kwargs={
            LoggerType.TRACKIO.value: {
                "name": get_run_name(
                    optimizer_fn=optimizer_fn, max_grad_norm=max_grad_norm
                ),
                "resume": "allow",
            }
        },
    )

    datasets = build_datasets(accelerator, root)
    num_classes = len(datasets.train.classes)

    metrics = build_metrics(num_classes)
    # Avoid accelerator.prepare() to prevent DDP sync conflicts.
    # TorchMetrics manages multi-GPU synchronization internally.
    metrics.train_loss.to(accelerator.device)
    metrics.train.to(accelerator.device)
    metrics.test.to(accelerator.device)

    # No grads/backward; saves VRAM
    eval_batch_size_mutable: list[int] = [int(initial_train_batch_size * 4)]

    @find_executable_batch_size(starting_batch_size=initial_train_batch_size)
    def _execute_train(train_batch_size: int):
        is_batch_size_saved = False
        accelerator.free_memory()
        set_seed(seed)

        model = ParaConv(in_channels=1, num_classes=num_classes)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optimizer_fn(params=model.parameters())
        train_loader = DataLoader(
            datasets.train,
            batch_size=train_batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=accelerator.device.type != "mps",
            shuffle=True,
        )

        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )

        for _ in tqdm(range(num_epochs)):
            epoch_results = train_one_epoch(
                accelerator=accelerator,
                dataloader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                metrics=metrics.train,
                train_loss_metric=metrics.train_loss,
                max_grad_norm=max_grad_norm,
            )
            if epoch_results is None:
                accelerator.log({"failure": True})
                break
            metrics_results, train_loss = epoch_results
            accelerator.log(
                {
                    "train/loss": train_loss.item(),
                    **{name: value.item() for name, value in metrics_results.items()},
                }
            )

            eval_results, eval_batch_size = evaluate(
                starting_eval_batch_size=eval_batch_size_mutable[0],
                accelerator=accelerator,
                test_dataset=datasets.test,
                model=model,
                metrics=metrics.test,
                num_workers=num_workers,
            )
            accelerator.log(
                {name: value.item() for name, value in eval_results.items()}
            )
            eval_batch_size_mutable[0] = eval_batch_size

            if not is_batch_size_saved:
                trackio_tracker: TrackioTracker = accelerator.get_tracker(
                    LoggerType.TRACKIO.value
                )
                trackio_tracker.store_init_configuration(
                    {
                        "train/batch_size": train_batch_size,
                        "val/batch_size": eval_batch_size_mutable[0],
                    }
                )
                is_batch_size_saved = True

            accelerator.save_state()

    _execute_train()
    accelerator.end_training()


optim_store = store(group="optimizer_fn")
optim_store(optim.SGD, zen_partial=True)
store(
    train_and_eval,
    name=train_and_eval.__name__,
    num_epochs=30,
    max_grad_norm=None,
    seed=28,
    hydra_defaults=[
        "_self_",
        {"optimizer_fn": optim.SGD.__name__},
    ],
)
if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(train_and_eval).hydra_main(
        config_name=train_and_eval.__name__, version_base=None
    )
