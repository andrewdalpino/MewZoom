from time import time

from argparse import ArgumentParser

import torch

from torch.nn.functional import interpolate
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available

from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.v2 import ToDtype
from torchvision.utils import make_grid, save_image

from src.mewzoom.model import MewZoom

import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser(
        description="Test and compare MewZoom upscaling with bicubic interpolation."
    )

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if "mps" in args.device and not mps_is_available():
        raise RuntimeError("MPS is not available.")

    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )

    upscaler = MewZoom(**checkpoint["upscaler_args"])

    upscaler.model.add_qa_head(checkpoint["degradation_features"])
    upscaler.model.add_weight_norms()

    state_dict = checkpoint["upscaler"]

    # Compensate for compiled state dict.
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)

    upscaler.load_state_dict(state_dict)

    upscaler.remove_parameterizations()
    upscaler.model.remove_qa_head()

    upscaler = upscaler.to(args.device)

    upscaler.eval()

    print("Model checkpoint loaded successfully")

    image_to_tensor = ToDtype(torch.float32, scale=True)

    image = decode_image(args.image_path, mode=ImageReadMode.RGB)

    x = image_to_tensor(image).unsqueeze(0).to(args.device)

    print("Upscaling ...")

    y_bicubic = interpolate(
        x,
        scale_factor=upscaler.model.upscale_ratio,
        mode="bicubic",
        align_corners=False,
        recompute_scale_factor=True,
    )

    y_pred = upscaler.upscale(x)

    # Remove batch dimension.
    y_bicubic = y_bicubic.squeeze(0)
    y_pred = y_pred.squeeze(0)

    pair = torch.stack([y_bicubic, y_pred], dim=0)

    grid = make_grid(pair, nrow=2)

    grid = grid.permute(1, 2, 0).to("cpu")

    plt.imshow(grid)
    plt.show()

    if "y" in input("Save images? (yes|no): ").lower():
        timestamp = time()

        save_image(y_bicubic, f"bicubic_{timestamp}.png")
        save_image(y_pred, f"y_pred_{timestamp}.png")


if __name__ == "__main__":
    main()
