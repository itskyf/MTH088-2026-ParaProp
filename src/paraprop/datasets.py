from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

from paraprop.typing import Split

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # `mean`: `FashionMNIST.data.float().mean()`
        # `std`: `FashionMNIST.data.float().std()`
        v2.Normalize(
            mean=(0.2860406015433517,), std=(0.35302426207299326,), inplace=True
        ),
    ]
)


def build_datasets(accelerator: Accelerator, root: Path) -> Split[Dataset]:
    if accelerator.is_main_process:
        FashionMNIST(root=root, train=True, download=True)
        FashionMNIST(root=root, train=False, download=True)
    accelerator.wait_for_everyone()

    train_dataset = FashionMNIST(root=root, train=True, transform=transform)
    test_dataset = FashionMNIST(root=root, train=False, transform=transform)
    return Split(train=train_dataset, test=test_dataset)
