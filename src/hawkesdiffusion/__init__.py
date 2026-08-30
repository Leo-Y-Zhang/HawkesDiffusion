# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Bivariate Hawkes modelling of information diffusion in event streams."""
from .hawkes import (
                     branching_matrix,
                     branching_ratio,
                     fit,
                     half_life,
                     log_likelihood,
                     rescaled_residuals,
                     simulate,
)

__all__ = ["branching_matrix", "branching_ratio", "fit", "half_life",
           "log_likelihood", "rescaled_residuals", "simulate"]
__version__ = "0.1.0"
