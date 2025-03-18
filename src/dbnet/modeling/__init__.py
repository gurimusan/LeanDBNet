import importlib

from src.dbnet.configs import Config
from .dbnet import DBNet

__all__ = ["build_model", "DBNet"]


def build_model(cfg: Config):
    mod = importlib.import_module("src.dbnet.modeling")
    klass = getattr(mod, cfg.model.model)
    instance = klass(**cfg.model.model_args)
    return instance
