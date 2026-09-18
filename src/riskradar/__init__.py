"""
RiskRadar — AI-powered women's safety risk intelligence platform.

Package layout
--------------
config    : central paths, constants, label schema
data      : dataset loading, validation, stratified splitting
features  : leakage-free domain feature engineering
train     : model benchmarking, tuning, ensembling, evaluation
explain   : SHAP-based global + local explainability
service   : inference service (singleton model holder)
schemas   : Pydantic request/response contracts
api       : FastAPI application
"""

__version__ = "2.0.0"
__author__ = "N. Venkatesan and Neethivendhan T."
__all__ = ["api", "config", "data", "explain", "features", "schemas", "service", "train"]
