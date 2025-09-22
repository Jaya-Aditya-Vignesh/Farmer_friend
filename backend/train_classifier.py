#!/usr/bin/env python3
"""
FIXED training script for ImageFolder structure
"""

import argparse
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import timm
import logging
from typing import Optional
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LitClassifier(pl.LightningModule):
    """Lightning classifier for ImageFolder pest detection"""

    def __init__(self,
                 model_name: str,
                 n_classes: int,
                 lr: float = 2e-4,
                 weight_decay: float = 1e-4,
                 class_weights: Optional[torch.Tensor] = None,
                 total_steps: int = 1000):  # Add total_steps parameter
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=n_classes)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Validation storage
        self.validation_step_outputs = []

        # Store total steps for scheduler
        self.total_steps = total_steps

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Compute accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()

        # Log every 100 steps
        if batch_idx % 100 == 0:
            self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=False)
            self.log('train/acc', acc, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Store for epoch end processing
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds.detach().cpu(),
            'targets': y.detach().cpu()
        })

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        if not outputs:
            return

        # Aggregate metrics
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # Compute accuracy
        acc = (all_preds == all_targets).float().mean()

        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val/acc', acc, prog_bar=True, sync_dist=True, on_epoch=True)

        # Clear memory
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.total_steps,  # Use stored total steps
            eta_min=self.hparams.lr * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


def main():
    parser = argparse.ArgumentParser(description="FIXED ImageFolder-based pest detection training")

    # Data arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing images/ folder with train/val/test subdirs")
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Training options
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--augmentation", type=str, default="medium",
                        choices=["light", "medium", "heavy"])
    parser.add_argument("--use_class_weights", action="store_true")

    # Hardware
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")

    # Output
    parser.add_argument("--out_dir", type=str, default="experiments/imagefolder")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seeds
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')

    logger.info("=" * 60)
    logger.info("FIXED ImageFolder Training Configuration:")
    logger.info(f"  Data root: {args.data_root}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Image size: {args.image_size}")
    logger.info(f"  Augmentation: {args.augmentation}")
    logger.info("=" * 60)

    # Import the updated data module - FIXED INHERITANCE
    from data import ImageFolderDataModule

    # Create PROPER Lightning data module
    class LightningImageFolderDataModule(pl.LightningDataModule):
        def __init__(self, base_dm):
            super().__init__()
            self.base_dm = base_dm
            self.n_classes = None
            self.class_to_idx = None

        def setup(self, stage=None):
            self.base_dm.setup()
            self.n_classes = self.base_dm.n_classes
            self.class_to_idx = self.base_dm.class_to_idx

        def train_dataloader(self):
            return self.base_dm.train_dataloader()

        def val_dataloader(self):
            return self.base_dm.val_dataloader()

        def test_dataloader(self):
            return self.base_dm.test_dataloader()

    # Create base data module
    base_dm = ImageFolderDataModule(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_level=args.augmentation,
        quick_validation=True
    )

    # Wrap in proper Lightning module
    dm = LightningImageFolderDataModule(base_dm)
    dm.setup()

    # Calculate total training steps
    train_loader = dm.train_dataloader()
    total_steps = len(train_loader) * args.epochs

    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights and dm.base_dm.train_ds:
        class_weights = dm.base_dm.train_ds.get_class_weights()
        logger.info("Using balanced class weights")

    # Create model with total steps
    model = LitClassifier(
        model_name=args.model,
        n_classes=dm.n_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        total_steps=total_steps  # Pass total steps
    )

    # Setup output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            filename="pest-{epoch:02d}-{val_loss:.3f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor="val/loss",
            patience=7,
            mode="min",
            min_delta=0.001
        )
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=str(out_dir), name="logs"),
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        benchmark=True
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, dm)

    # Test if test set available
    try:
        if hasattr(dm.base_dm, 'test_ds') and dm.base_dm.test_ds:
            logger.info("Running test evaluation...")
            trainer.test(model, dm, ckpt_path='best')
    except:
        logger.info("Test evaluation skipped - no test set or error occurred")

    # Save training info
    info = {
        'model_name': args.model,
        'n_classes': dm.n_classes,
        'class_to_idx': dm.class_to_idx,
        'train_samples': len(dm.base_dm.train_ds) if dm.base_dm.train_ds else 0,
        'val_samples': len(dm.base_dm.val_ds) if dm.base_dm.val_ds else 0,
        'test_samples': len(dm.base_dm.test_ds) if dm.base_dm.test_ds else 0,
        'config': vars(args)
    }

    with open(out_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"Training complete! Results saved to {out_dir}")


if __name__ == "__main__":
    main()
