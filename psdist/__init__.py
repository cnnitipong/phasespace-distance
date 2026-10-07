"""psdist: compare two signals in reconstructed phase space (CPST and percentile Hausdorff)."""
from .core import (embed, average_mutual_information, first_minimum,
                   false_nearest_neighbours, cpst, hausdorff, compare)
from . import mock  # noqa: F401

__version__ = "1.1.0"
__all__ = ["embed", "average_mutual_information", "first_minimum",
           "false_nearest_neighbours", "cpst", "hausdorff", "compare"]
