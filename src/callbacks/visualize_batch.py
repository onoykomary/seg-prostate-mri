import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning.pytorch as pl


class VisBatchCallback(pl.Callback):
    def __init__(self, save_dir="logs/augmentations", num_batches_to_check=3):
        super().__init__()
        self.save_dir = save_dir
        self.num_batches_to_check = num_batches_to_check
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch != 0 or batch_idx >= self.num_batches_to_check:
            return

        # Images: [B, C, H, W, D], Labels: [B, 1, H, W, D]
        images = batch["image"].detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()

        batch_size = images.shape[0]
        num_slices = 3

        fig, axes = plt.subplots(batch_size, num_slices, figsize=(num_slices * 4, batch_size * 4))

        if batch_size == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(batch_size):
            img_vol = images[i, 0]
            lbl_vol = labels[i, 0]

            depth = img_vol.shape[-1]
            slices_to_plot = [depth // 4, depth // 2, (3 * depth) // 4]

            for col_idx, z_idx in enumerate(slices_to_plot):
                ax = axes[i, col_idx]

                img_slice = img_vol[:, :, z_idx]
                lbl_slice = lbl_vol[:, :, z_idx]

                ax.imshow(img_slice.T, cmap="gray", origin="lower")

                if lbl_slice.sum() > 0:
                    masked_lbl = np.ma.masked_where(lbl_slice == 0, lbl_slice)

                    ax.imshow(
                        masked_lbl.T, cmap="autumn", alpha=0.5, vmin=0, vmax=1, origin="lower"
                    )

                ax.set_title(f"Sample {i} | Z={z_idx}")
                ax.axis("off")

        plt.suptitle(f"Batch: {batch_idx} (Epoch {trainer.current_epoch})", fontsize=16)
        plt.tight_layout()

        save_path = os.path.join(
            self.save_dir, f"batch_{batch_idx}_epoch{trainer.current_epoch}.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
