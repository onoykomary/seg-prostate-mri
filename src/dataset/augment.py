import monai.transforms as mt


def get_transforms(mode, patch_size=None, num_samples=2):
    # 1. --- Base transforms
    transforms = [
        mt.LoadImaged(keys=["image", "label"]),
        mt.EnsureChannelFirstd(keys=["image", "label"]),
        mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
        mt.CropForegroundd(
            keys=["image", "label"],
            source_key="image",
        ),
        mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]

    if mode == "train":
        transforms.extend(
            [
                # 2. --- Work with patches
                mt.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
                mt.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=patch_size,
                    pos=3,
                    neg=1,
                    num_samples=num_samples,
                ),
                # 3. --- MRI Augmentation
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                mt.RandRotated(
                    keys=["image", "label"],
                    range_x=0.2,
                    range_y=0.2,
                    range_z=0.2,
                    prob=0.2,
                    mode=["bilinear", "nearest"],
                ),
                mt.RandGaussianNoised(keys="image", prob=0.4, mean=0.0, std=0.1),
                mt.RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.5)),
                mt.RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
            ]
        )
    return mt.Compose(transforms)
