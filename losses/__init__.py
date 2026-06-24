"""
losses/__init__.py

Public API for the losses package.
"""

from losses.hydro_loss import HydrologicalConsistencyLoss
from losses.combined_loss import BCEDiceHydroLoss

__all__ = ["HydrologicalConsistencyLoss", "BCEDiceHydroLoss"]