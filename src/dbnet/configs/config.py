from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

from dataclass_wizard import JSONPyWizard


class OptimizerType(StrEnum):
    SGD = "sgd"
    ADAM = "adam"


class SchedulerType(StrEnum):
    COSINE = "cosine"
    ONECYCLE = "onecycle"
    POLY = "poly"


@dataclass
class ModelConfig:
    model: str = "DBNet"
    model_args: dict[str, Any] = field(default_factory=lambda: {
        "backbone": "resnet18",  # Paper default, change to "deformable_resnet50" for better performance
        "backbone_args": {},
        "decoder": "DBNetDecoder",
        "decoder_args": {},
        "loss": "DBNetLoss",
        "loss_args": {},
        })


@dataclass
class TrainingConfig:
    workers: int | None = None
    epochs: int = 10
    batch_size: int = 2
    dataset_class: str = "ICDAR2015Dataset"
    img_root: list[str] = field(default_factory=list)
    gt_root: list[str] = field(default_factory=list)
    val_ratio: float = 0.2
    scheduler: SchedulerType = SchedulerType.POLY
    optimizer: OptimizerType = OptimizerType.ADAM
    lr: float = 0.007
    weight_decay: float = 0


@dataclass
class ValidationConfig:
    workers: int | None = None
    batch_size: int = 8
    dataset_class: str = "ICDAR2015Dataset"
    img_root: list[str] = field(default_factory=list)
    gt_root: list[str] = field(default_factory=list)


@dataclass
class Config(JSONPyWizard):
    # Config for model option
    model: ModelConfig

    # Config for training option
    training: TrainingConfig

    # Config for evaluation option
    validation: ValidationConfig

    log_dir: str = "logs"

    config: Optional[str] = None
