from __future__ import annotations

from enum import StrEnum

from dataclasses import dataclass, field

from dataclass_wizard import JSONPyWizard

class OptimizerType(StrEnum):
    ADAM = "adam"
    ADAMW = "adamw"


class SchedulerType(StrEnum):
    COSINE = "cosine"
    ONECYCLE = "onecycle"
    POLY = "poly"


@dataclass
class ModelConfig:
    model: str = "DBNet"
    model_args: dict[str, any] = field(default_factory=lambda: {
        "backbone": "deformable_resnet50",
        "backbone_args": {},
        "decoder": "DBNetDecoder",
        "decoder_args": {},
        "loss": "DBNetLoss",
        "loss_args": {},
        })


@dataclass
class TrainingConfig:
    workers: int | None = None
    batch_size: int = 8
    epochs: int = 10
    dataset_class: str = "ICDAR2015Dataset"
    img_root: list[str] = field(default_factory=lambda: [])
    gt_root: list[str] = field(default_factory=lambda: [])
    val_ratio: float = 0.2
    lr: float = 1e-3
    scheduler: SchedulerType = SchedulerType.POLY
    optimizer: OptimizerType = OptimizerType.ADAM
    weight_decay: float = 0


@dataclass
class ValidationConfig:
    workers: int | None = None
    batch_size: int = 8
    dataset_class: str = "ICDAR2015Dataset"
    img_root: list[str] = field(default_factory=lambda: [])
    gt_root: list[str] = field(default_factory=lambda: [])


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
