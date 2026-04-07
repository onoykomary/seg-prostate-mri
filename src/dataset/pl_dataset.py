import lightning.pytorch as pl

from monai.data import CacheDataset
from monai.data import DataLoader, list_data_collate
import glob
import os

from src.dataset.augment import get_transforms

class ProstateDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["data"]["path"]

    def _get_files(self, folder_name):
        images = sorted(glob.glob(os.path.join(self.data_dir, f"images/{folder_name}/*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(self.data_dir, f"labels/{folder_name}/*.nii.gz")))

        return [{"image": i, "label": l} for i, l in zip(images, labels)]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_files = self._get_files("train")
            self.train_ds = CacheDataset(
                data=train_files,
                transform=get_transforms(
                    "train",
                    self.config["training"]["patch_size"],
                    self.config["training"]["num_samples"],
                ),
                cache_rate=1.0,
                num_workers=self.config["training"]["num_workers"],
            )

        if stage == "fit" or stage == "validate" or stage is None:
            val_files = self._get_files("val")
            self.val_ds = CacheDataset(
                data=val_files,
                transform=get_transforms("val"),
                cache_rate=1.0,
                num_workers=self.config["training"]["num_workers"],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True,
            collate_fn=list_data_collate,
            persistent_workers=self.config["training"]["num_workers"] > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config["training"]["val_batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True,
            collate_fn=list_data_collate,
            persistent_workers=self.config["training"]["num_workers"] > 0,
        )
