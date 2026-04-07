import yaml
import torch
import os
import time
import shutil
from monai.utils import set_determinism
import lightning.pytorch as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from src.callbacks.visualize_batch import VisBatchCallback
from src.dataset.pl_dataset import ProstateDataModule
from src.train.pl_wrap import TrainPipeline

config_path = "configs/config.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

# ---- SET SEEDS
SEED = config["seed"]
pl.seed_everything(SEED, workers=True)
set_determinism(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- EXPERIMENT NAME
experiment_name = f"prostate_{time.strftime('%Y%m%d_%H%M%S')}"

# ---- CALLBACKS
monitor = config["logging"]["checkpoint_monitor"]
mode = config["logging"]["checkpoint_mode"]
patience = config["training"]["earlystop"]

checkpoint_callback = ModelCheckpoint(
    monitor=monitor,
    mode=mode,
    save_top_k=config["logging"]["save_top_k"],
    filename="{epoch:02d}-{val_dice:.4f}",
    dirpath=f"Experiments/{experiment_name}",
)

lr_monitor = LearningRateMonitor(logging_interval="step")
vis_callback = VisBatchCallback(
    save_dir=f"Experiments/{experiment_name}/visualizations", num_batches_to_check=3
)
early_stop = EarlyStopping(monitor=monitor, patience=patience, mode=mode)

logger = None
if config["logging"]["mlflow"]:
    logger = MLFlowLogger(
        experiment_name="prostate_segmentation",
        run_name=f"UNet_LR_{config['optimizer']['optimizer_params']['lr']}",
        tracking_uri="file:./mlruns",
    )

# ---- TRAINING STEP
trainer = pl.Trainer(
    max_epochs=config["training"]["max_epochs"],
    accelerator=config["training"]["accelerator"],
    devices=1,
    precision=config["training"]["precision"],
    accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
    gradient_clip_val=config["training"]["gradient_clip_val"],
    callbacks=[checkpoint_callback, lr_monitor, vis_callback, early_stop],
    logger=logger,
    log_every_n_steps=10,
)

model = TrainPipeline(config)
datamodule = ProstateDataModule(config)
trainer.fit(model=model, datamodule=datamodule)

# --- COPY YAML
shutil.copy2(config_path, f'{os.path.join("Experiments", experiment_name, "classify.yaml")}')
