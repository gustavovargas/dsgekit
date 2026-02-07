"""Filtering, forecasting, and decomposition."""

from dsgekit.filters.forecast import (
    ForecastResult,
    HistoricalDecompositionResult,
    forecast,
    historical_decomposition,
)
from dsgekit.filters.kalman import (
    KalmanResult,
    kalman_filter,
    log_likelihood,
)
from dsgekit.filters.smoother import (
    SmootherResult,
    kalman_smoother,
)

__all__ = [
    "KalmanResult",
    "kalman_filter",
    "log_likelihood",
    "SmootherResult",
    "kalman_smoother",
    "ForecastResult",
    "HistoricalDecompositionResult",
    "forecast",
    "historical_decomposition",
]
