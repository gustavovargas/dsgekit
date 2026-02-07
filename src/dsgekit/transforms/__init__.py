"""Transformations: linearization, state-space conversion, normalization."""

from dsgekit.transforms.linearize import (
    LinearizedSystem,
    check_linearization,
    linearize,
)
from dsgekit.transforms.statespace import (
    StateSpace,
    to_state_space,
    validate_state_space,
)

__all__ = [
    "LinearizedSystem",
    "linearize",
    "check_linearization",
    "StateSpace",
    "to_state_space",
    "validate_state_space",
]
