from pathlib import Path

import numpy as np

import torch
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR, PolynomialLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchinfo import summary

from tqdm.auto import tqdm

from src.dbnet.modeling.backbones import deformable_resnet50, deformable_resnet18, ResNet18_Weights
from src.dbnet.configs import Config, ModelConfig, TrainingConfig, ValidationConfig
from src.dbnet.datasets import ICDAR_MEAN, ICDAR_STD, ICDAR2015Dataset, collate_fn
from src.dbnet.datasets import transforms as T
from src.dbnet.modeling import build_model
from src.dbnet.metrics.icdar2015 import QuadMetric
from src.dbnet.utils import EarlyStopper


def train(model, train_loader, optimizer, scheduler, batch_transforms):
    model.train()

    train_loss, batch_cnt = 0, 0
    pbar = tqdm(train_loader, dynamic_ncols=True)
    for images, targets in pbar:
        images, targets = batch_transforms(images, targets)

        images = images.cuda()

        optimizer.zero_grad()

        train_loss = model(images, targets)["loss"]
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        pbar.set_description(f"Training loss: {train_loss.item():.6} | LR: {last_lr:.6}")

        train_loss += train_loss.item()
        batch_cnt += 1

    train_loss /= batch_cnt
    return train_loss, last_lr


@torch.no_grad()
def evaluate(model, val_loader, val_metric, batch_transforms):
    model.eval()

    val_loss, batch_cnt = 0, 0
    raw_metrics = []
    pbar = tqdm(val_loader, dynamic_ncols=True)
    for images, targets in pbar:
        images, targets = batch_transforms(images, targets)

        images = images.cuda()

        out = model(images, targets, return_preds=True)

        # Compute metric
        loc_preds = out["preds"]
        raw_metric = val_metric.validate_measure(
            images, targets, loc_preds)
        raw_metrics.append(raw_metric)

        pbar.set_description(f"Validation loss: {out['loss'].item():.6}")

        val_loss += out["loss"].item()
        batch_cnt += 1

    recall, precision, fmeasure = val_metric.gather_measure(raw_metrics)
    val_loss /= batch_cnt
    return val_loss, recall, precision, fmeasure


def get_dataset(args, c):
    train_set = ICDAR2015Dataset(
        img_root=c.training.img_root,
        gt_root=c.training.gt_root,
        transforms=T.Compose([
            T.RandomApply([T.HorizontalFlip()], p=0.25),
            T.RandomApply([T.RandomRotate((-10, 10))], p=0.25),
            T.RandomChoice([
                T.RandomApply([T.RandomCrop(size=(640, 640), max_tries=10)], p=0.15),
                T.RandomApply([T.RandomMinResize((480, 512, 544, 576, 608), 1024)], p=0.15),
                ]),
            ]),
        )
    val_set = ICDAR2015Dataset(
        img_root=c.validation.img_root,
        gt_root=c.validation.gt_root,
        )
    return train_set, val_set


def main(args):
    pbar = tqdm()

    c = Config(
        model=ModelConfig(
            model_args={
                "backbone": args.backbone,
                "decoder": "DBNetDecoder",
                "loss": "DBNetLoss",
                }),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_root=[
                "./datasets/icdar2015/ch4_training_images",
                ],
            gt_root=[
                "./datasets/icdar2015/ch4_training_localization_transcription_gt",
                ],
            ),
        validation=ValidationConfig(
            batch_size=args.batch_size,
            img_root=[
                "./datasets/icdar2015/ch4_test_images",
                ],
            gt_root=[
                "./datasets/icdar2015/Challenge4_Test_Task1_GT",
                ],
            ),
        )

    # dataset
    train_set, val_set = get_dataset(args, c)

    # loader
    train_loader = DataLoader(
        train_set,
        batch_size=c.training.batch_size,
        drop_last=True,
        num_workers=c.training.workers if c.training.workers else 0,
        sampler=RandomSampler(train_set),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=c.validation.batch_size,
        drop_last=False,
        num_workers=c.validation.workers if c.validation.workers else 0,
        sampler=SequentialSampler(val_set),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    # Load model
    model = build_model(c)
    model = model.cuda()
    model.train()

    # Resume weights
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

    # optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        c.training.lr,
        betas=(0.95, 0.999),
        eps=1e-6,
        weight_decay=0,
    )

    # scheduler
    scheduler = PolynomialLR(optimizer, c.training.epochs * len(train_loader))

    # metrics
    val_metric = QuadMetric()

    # stopper
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)

    # batch transforms
    batch_transforms = T.Normalize(mean=ICDAR_MEAN, std=ICDAR_STD)

    exp_name = args.name

    min_loss = np.inf
    for epoch in range(c.training.epochs):
        train_loss, actual_lr = train(
            model, train_loader, optimizer, scheduler, batch_transforms,
        )

        val_loss, recall, precision, fmeasure = evaluate(
            model, val_loader, val_metric, batch_transforms,
        )

        if args.save_only_loss_decreased:
            if val_loss < min_loss:
                pbar.write(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
                torch.save(model.state_dict(), Path(args.output_dir) / f"{exp_name}.pt")
                min_loss = val_loss
        else:
            pbar.write(f"Saving state...")
            torch.save(model.state_dict(), Path(args.output_dir) / f"{exp_name}.pt")

        log_msg = f"Epoch {epoch + 1}/{c.training.epochs} - Validation loss: {val_loss:.6} "
        if any(val is None for val in (recall, precision, fmeasure)):
            log_msg += "(Undefined metric value, caused by empty GTs or predictions)"
        else:
            log_msg += f"(Recall: {recall.val:.2%} ({recall.count}) | Precision: {precision.val:.2%} ({precision.count}) | F-measure: {fmeasure.val:.2%} ({fmeasure.count}) )"
        pbar.write(log_msg)


        if args.early_stop and early_stopper.early_stop(val_loss):
            pbar.write("Training halted early due to reaching patience limit.")
            break


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--name", type=str, default="dbnet_resnet18", help="name of your training experiment")
    parser.add_argument("--output-dir", type=str, default="out", help="path to save checkpoints and final model")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="batch size for training")
    parser.add_argument("--backbone", type=str, default="deformable_resnet18", help="backbone model")
    parser.add_argument("--dataset", type=str, default="icdar2015", help="dataset class")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-epochs", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--early-stop-delta", type=float, default=0.01, help="Minimum Delta for early stopping")
    parser.add_argument("--save-only-loss-decreased", action="store_true", help="save only loss decreased")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
