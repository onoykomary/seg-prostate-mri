from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import os

RAW_DIR = Path("data/raw")
OUT_IMG_DIR = Path("data/processed/images")
OUT_LBL_DIR = Path("data/processed/labels")

TARGET_SPACING = (1.0, 1.0, 1.0)


# -----  Resample image
def resample(image, spacing, is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


# ----- N4 Bias Field Correction
def bias_correction(image):
    image = sitk.Cast(image, sitk.sitkFloat32)

    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    # insideValue, outsideValue, numberOfHistogramBins

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30])
    corrector.SetConvergenceThreshold(1e-6)

    corrected = corrector.Execute(image, mask)
    return corrected


# ----- crop foreground
def crop_foreground(image, label):
    arr = sitk.GetArrayFromImage(image)
    foreground = arr > np.percentile(arr, 1)

    coords = np.where(foreground)
    zmin, ymin, xmin = np.min(coords, axis=1)
    zmax, ymax, xmax = np.max(coords, axis=1)

    size = [
        int(xmax - xmin + 1),
        int(ymax - ymin + 1),
        int(zmax - zmin + 1),
    ]
    index = [int(xmin), int(ymin), int(zmin)]

    image = sitk.RegionOfInterest(image, size, index)
    label = sitk.RegionOfInterest(label, size, index)

    return image, label


def main():
    for split in ["train", "val"]:
        (OUT_IMG_DIR / split).mkdir(parents=True, exist_ok=True)
        (OUT_LBL_DIR / split).mkdir(parents=True, exist_ok=True)

        label_files = sorted((RAW_DIR / split).rglob("*_segmentation.mhd"))

        label_map = {p.stem.replace("_segmentation", ""): p for p in label_files}

        for pid, lbl_path in tqdm(label_map.items(), total=len(label_map)):
            img_path = lbl_path.with_name(f"{pid}.mhd")

            if not img_path.exists():
                print(f"WARNING! Image not found: {img_path}")
                continue

            img = sitk.ReadImage(str(img_path))
            lbl = sitk.ReadImage(str(lbl_path))

            # resample
            img = resample(img, TARGET_SPACING, is_label=False)
            lbl = resample(lbl, TARGET_SPACING, is_label=True)

            # bias field
            img = bias_correction(img)

            # crop
            img, lbl = crop_foreground(img, lbl)

            # save
            split = img_path.parent.name
            sitk.WriteImage(img, OUT_IMG_DIR / split / f"{pid}_image.nii.gz")
            sitk.WriteImage(lbl, OUT_LBL_DIR / split / f"{pid}_label.nii.gz")


if __name__ == "__main__":
    main()
