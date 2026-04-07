import os
import numpy as np
import torch
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
    KeepLargestConnectedComponent,
    FillHoles,
)
from monai.data import decollate_batch


class TrainPipeline(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.save_hyperparameters(config)

        # model
        self.model = UNet(
            spatial_dims=config["model"]["spatial_dims"],
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            channels=config["model"]["channels"],
            strides=config["model"]["strides"],
            norm=config["model"]["norm"],
            num_res_units=config["model"]["res_units"],
        )

        # metrics
        self.train_metrics = DiceMetric(include_background=False)
        self.val_metrics = DiceMetric(include_background=False)
        self.hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
        self.surf_dist = SurfaceDistanceMetric(include_background=False, symmetric=True)

        # loss
        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True, batch=True)

        # postprocessing
        self.post_label = Compose(
            [
                EnsureType("tensor"),
                AsDiscrete(to_onehot=2),
            ]
        )

        self.post_pred = Compose(
            [
                EnsureType("tensor"),
                AsDiscrete(argmax=True, to_onehot=2),
                KeepLargestConnectedComponent(applied_labels=[1]),
                FillHoles(applied_labels=[1]),
                EnsureType("tensor"),
            ]
        )

        self.simplified_post_pred = Compose(
            [EnsureType("tensor"), AsDiscrete(argmax=True, to_onehot=2)]
        )

    def configure_optimizers(self):
        # AdamW optimizer
        if self.config["optimizer"]["type"] == "AdamW":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config["optimizer"]["optimizer_params"],
            )
        # SGD optimizer
        elif self.config["optimizer"]["type"] == "SGD":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                momentum=0.9,
                nesterov=True,
                **self.config["optimizer"]["optimizer_params"],
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.config['optimizer']}")

        scheduler_cfg = self.hparams["scheduler"]
        if not scheduler_cfg or not scheduler_cfg.get("type"):
            return optimizer

        if scheduler_cfg["type"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                **scheduler_cfg["scheduler_params"],
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_dice",
                },
            }
        # cosine + warmup optimizer
        elif scheduler_cfg["type"] == "cosine":
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            total_training_steps = steps_per_epoch * self.config["training"]["max_epochs"]

            warmup_steps = steps_per_epoch * self.config["scheduler"].get("warmup_epochs", 0.1)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        raise ValueError(f"Unknown scheduler type: {scheduler_cfg['type']}")

    def infer_batch(self, batch):
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        return logits, y

    def training_step(self, batch, batch_idx):
        logits, y = self.infer_batch(batch)

        y = y.long()

        loss = self.criterion(logits, y)

        preds = [self.simplified_post_pred(i) for i in decollate_batch(logits)]
        y = [self.post_label(i) for i in decollate_batch(y)]

        self.train_metrics(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        dice = self.train_metrics.aggregate().item()
        self.train_metrics.reset()
        self.log("train_dice", dice, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        logits = sliding_window_inference(
            inputs=x,
            roi_size=self.config["training"]["roi_size"],
            sw_batch_size=self.config["training"]["sw_batch_size"],
            predictor=self.model,
            overlap=self.config["training"]["overlap"],
            mode="gaussian",
        )

        loss = self.criterion(logits, y)

        preds = [self.post_pred(i) for i in decollate_batch(logits)]
        gts = [self.post_label(i) for i in decollate_batch(y)]

        self.val_metrics(y_pred=preds, y=gts)
        self.hd95_metric(y_pred=preds, y=gts)
        self.surf_dist(y_pred=preds, y=gts)

        # --- volume error calculation
        pixdim = batch["image"].meta["pixdim"][:, 1:4]

        batch_vol_errors = []
        for i in range(len(preds)):
            voxel_vol = torch.prod(pixdim[i]).item()

            pred_voxels = torch.sum(preds[i][1] > 0.5).item()
            gt_voxels = torch.sum(gts[i][1] > 0.5).item()

            vol_pred = (pred_voxels * voxel_vol) / 1000.0
            vol_gt = (gt_voxels * voxel_vol) / 1000.0

            batch_vol_errors.append(abs(vol_pred - vol_gt))

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_vol_err", np.mean(batch_vol_errors), on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        dice = self.val_metrics.aggregate().item()
        hd95 = self.hd95_metric.aggregate().item()
        surf_dist = self.surf_dist.aggregate().item()

        self.val_metrics.reset()
        self.hd95_metric.reset()
        self.surf_dist.reset()

        self.log("val_dice", dice, prog_bar=True, on_epoch=True)
        self.log("val_hd95", hd95, prog_bar=True, on_epoch=True)
        self.log("surf_dist", surf_dist, prog_bar=True, on_epoch=True)
