from time import time

from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)

from src.ultrazoom.model import MewZoom
from src.ultrazoom.control import ControlVector

from data import ImagePairs

from tqdm import tqdm


def main():
    parser = ArgumentParser(
        description="Single-image super-resolution validation script"
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--lr_images_path", default="./dataset/validate/lr", type=str)
    parser.add_argument("--hr_images_path", default="./dataset/validate/hr", type=str)
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")

    dataset = ImagePairs(args.lr_images_path, args.hr_images_path)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory="cuda" in args.device,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)

    model = MewZoom(**checkpoint["model_args"])

    model.add_qa_head(checkpoint["degradation_features"])
    model.add_weight_norms()

    state_dict = checkpoint["model"]

    # Compensate for compiled state dict.
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    model.remove_parameterizations()
    model.remove_qa_head()

    model = model.to(args.device)

    model.eval()

    print("Model checkpoint loaded successfully")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    vif_metric = VisualInformationFidelity().to(args.device)

    for x, y in tqdm(dataloader, desc="Testing", leave=False):
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)

        y_pred = model.upscale(x)

        psnr_metric.update(y_pred, y)
        ssim_metric.update(y_pred, y)
        vif_metric.update(y_pred, y)

    psnr = psnr_metric.compute()
    ssim = ssim_metric.compute()
    vif = vif_metric.compute()

    print(f"PSNR: {psnr:.5f}, SSIM: {ssim:.5f}, VIF: {vif:.5f}")


if __name__ == "__main__":
    main()
