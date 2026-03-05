import sys
from functools import partial
from typing import NamedTuple

import accelerate.utils


class Split[T](NamedTuple):
    train: T
    test: T


tqdm = partial(
    accelerate.utils.tqdm, dynamic_ncols=True, disable=not sys.stdout.isatty()
)
