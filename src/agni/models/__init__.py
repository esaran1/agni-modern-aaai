from agni.models.logreg import LogisticRegressionModel
from agni.models.propensity_severity import NaiveSeverityModel, PropensityWeightedSeverityModel
from agni.models.random_forest import RandomForestModel
from agni.models.transformer import TransformerModel
from agni.models.xgboost import XGBoostModel


def build_model(name: str, config: dict):
    registry = {
        "xgboost": XGBoostModel,
        "random_forest": RandomForestModel,
        "logreg": LogisticRegressionModel,
        "transformer": TransformerModel,
        "naive_severity": NaiveSeverityModel,
        "propensity_severity": PropensityWeightedSeverityModel,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'")
    return registry[name](config)


__all__ = [
    "LogisticRegressionModel",
    "NaiveSeverityModel",
    "PropensityWeightedSeverityModel",
    "RandomForestModel",
    "TransformerModel",
    "XGBoostModel",
    "build_model",
]
