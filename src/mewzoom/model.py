from math import floor, ceil
from abc import ABC, abstractmethod

from typing import Self

from functools import partial

import torch

from torch import Size, Tensor

from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    SiLU,
    Sigmoid,
    PixelShuffle,
    AdaptiveAvgPool2d,
    Flatten,
)

from torch.nn.functional import pad

from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin


type FeatureMapSize = Size | tuple[int, int] | list[int]


class Upscaler(ABC):
    """An interface for upscaler models."""

    @abstractmethod
    def upscale(self, x: Tensor) -> Tensor:
        pass


class Critic(ABC):
    """An interface for critic networks."""

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        pass


class MewZoomTrunkNet(Upscaler, Module):
    """
    A fast single-image super-resolution model with a deep low-resolution encoder network
    and high-resolution sub-pixel convolutional decoder head.
    """

    AVAILABLE_UPSCALE_RATIOS = {2, 3, 4, 8}

    def __init__(
        self,
        upscale_ratio: int,
        num_channels: int,
        num_layers: int,
        hidden_ratio: int,
        exciter_hidden_ratio: int,
    ):
        super().__init__()

        assert (
            upscale_ratio in self.AVAILABLE_UPSCALE_RATIOS
        ), f"Upscale ratio must be one of {self.AVAILABLE_UPSCALE_RATIOS}, but got {upscale_ratio}."

        self.stem = FanOutProjection(3, num_channels)

        self.body = TrunkNet(
            num_channels, num_layers, hidden_ratio, exciter_hidden_ratio
        )

        self.head = SuperResolver(num_channels, upscale_ratio)

        self.upscale_ratio = upscale_ratio

    @property
    def num_params(self) -> int:
        """Total number of parameters in the model."""

        return sum(param.numel() for param in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def initialize_weights(self) -> None:
        """Initialize all model weights using Kaiming uniform initialization."""

        self.stem.initialize_weights()
        self.body.initialize_weights()
        self.head.initialize_weights()

    def add_qa_head(self, num_features: int) -> None:
        """Add a quality assessment head to predict degradation features."""

        self.body.add_qa_head(num_features)

    def remove_qa_head(self) -> None:
        """Remove the quality assessment head."""

        self.body.remove_qa_head()

    def add_weight_norms(self) -> None:
        """Add weight normalization parameterization to the network."""

        self.stem.add_weight_norms()
        self.body.add_weight_norms()
        self.head.add_weight_norms()

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.body.enable_activation_checkpointing()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).
        """

        z = self.stem.forward(x)
        z, z_qa = self.body.forward(z)
        z = self.head.forward(z)

        return z, z_qa

    @torch.inference_mode()
    def upscale(self, x: Tensor) -> Tensor:
        """
        Convenience method for inference.

        Args:
            x: Input image tensor of shape (B, 3, H, W).
        """

        z, _ = self.forward(x)

        z = torch.clamp(z, 0, 1)

        return z


class MewZoomUNet(Upscaler, Module):
    """
    A model for image super-resolution based on a U-Net.
    """

    AVAILABLE_UPSCALE_RATIOS = {2, 3, 4, 8}

    def __init__(
        self,
        upscale_ratio: int,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
        exciter_hidden_ratio: int,
    ):
        super().__init__()

        assert (
            upscale_ratio in self.AVAILABLE_UPSCALE_RATIOS
        ), f"Upscale ratio must be one of {self.AVAILABLE_UPSCALE_RATIOS}, but got {upscale_ratio}."

        self.stem = FanOutProjection(3, primary_channels)

        self.body = UNet(
            primary_channels,
            primary_layers,
            secondary_channels,
            secondary_layers,
            tertiary_channels,
            tertiary_layers,
            quaternary_channels,
            quaternary_layers,
            hidden_ratio,
            exciter_hidden_ratio,
        )

        self.head = SuperResolver(primary_channels, upscale_ratio)

        self.upscale_ratio = upscale_ratio

    @property
    def num_params(self) -> int:
        """Total number of parameters in the model."""

        return sum(param.numel() for param in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def initialize_weights(self) -> None:
        """Initialize all model weights using Kaiming uniform initialization."""

        self.stem.initialize_weights()
        self.body.initialize_weights()
        self.head.initialize_weights()

    def add_qa_head(self, num_features: int) -> None:
        """Add a quality assessment head to predict degradation features."""

        self.body.add_qa_head(num_features)

    def remove_qa_head(self) -> None:
        """Remove the quality assessment head."""

        self.body.remove_qa_head()

    def add_weight_norms(self) -> None:
        """Add weight normalization parameterization to the network."""

        self.stem.add_weight_norms()
        self.body.add_weight_norms()
        self.head.add_weight_norms()

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder and decoder block.
        """

        self.body.enable_activation_checkpointing()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).

        """

        z = self.stem.forward(x)
        z, z_qa = self.body.forward(z)
        z = self.head.forward(z)

        return z, z_qa

    @torch.inference_mode()
    def upscale(self, x: Tensor) -> Tensor:
        """
        Convenience method for inference.

        Args:
            x: Input image tensor of shape (B, 3, H, W).
        """

        z, _ = self.forward(x)

        z = torch.clamp(z, 0, 1)

        return z


class MewZoom(Upscaler, Module, PyTorchModelHubMixin):
    """
    A factory class for MewZoom super-resolution models.

    Example usage:

        model = MewZoom(
            architecture="trunknet",
            upscale_ratio=2,
            num_channels=64,
            num_layers=8,
            hidden_ratio=2,
            exciter_hidden_ratio=4,
        )

        model = MewZoom(
            architecture="unet",
            upscale_ratio=2,
            primary_channels=64,
            primary_layers=4,
            secondary_channels=128,
            secondary_layers=4,
            tertiary_channels=256,
            tertiary_layers=4,
            quaternary_channels=512,
            quaternary_layers=2,
            hidden_ratio=2,
            exciter_hidden_ratio=4,
        )
    """

    def __init__(self, architecture: str, **kwargs):
        super().__init__()

        match architecture.lower():
            case "trunknet":
                model = MewZoomTrunkNet(**kwargs)

            case "unet":
                model = MewZoomUNet(**kwargs)

            case _:
                raise ValueError("Invalid model architecture.")

        self.model = model

    def initialize_weights(self) -> None:
        self.model.initialize_weights()

    def remove_parameterizations(self) -> None:
        """Remove all network parameterizations."""

        for module in self.modules():
            if is_parametrized(module):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).
        """

        return self.model.forward(x)

    @torch.inference_mode()
    def upscale(self, x: Tensor) -> Tensor:
        """
        Convenience method for inference.

        Args:
            x: Input image tensor of shape (B, 3, H, W).
        """

        return self.model.upscale(x)


class ONNXModel(Module):
    """A wrapper class for exporting to ONNX format."""

    def __init__(self, model: Upscaler):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).
        """

        return self.model.upscale(x)


class FanOutProjection(Module):
    """A linear projection for expanding the number of channels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        assert in_channels > 0, "Input channels must be greater than 0."

        assert (
            in_channels < out_channels
        ), "Output channels must be greater than input channels."

        self.conv = Conv2d(in_channels, out_channels, kernel_size=1)

    def initialize_weights(self) -> None:
        self.conv.reset_parameters()

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def add_spectral_norms(self, num_iterations: int) -> None:
        self.conv = spectral_norm(self.conv, n_power_iterations=num_iterations)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)

        return z


class TrunkNet(Module):
    """A low-resolution scale subnetwork employing a deep stack of encoder blocks."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        hidden_ratio: int,
        exciter_hidden_ratio: int,
    ):
        super().__init__()

        assert num_layers >= 4, "Number of layers must be greater than or equal to 4."

        new_encoder_block = partial(
            EncoderBlock,
            hidden_ratio=hidden_ratio,
            exciter_hidden_ratio=exciter_hidden_ratio,
        )

        self.stage1 = Sequential(
            *[new_encoder_block(num_channels) for _ in range(ceil(num_layers / 4))]
        )

        self.stage2 = Sequential(
            *[new_encoder_block(num_channels) for _ in range(floor(num_layers / 4))]
        )

        self.stage3 = Sequential(
            *[new_encoder_block(num_channels) for _ in range(ceil(num_layers / 4))]
        )

        self.stage4 = Sequential(
            *[new_encoder_block(num_channels) for _ in range(floor(num_layers / 4))]
        )

        self.skip1 = GatedSkipConnection(num_channels, exciter_hidden_ratio)
        self.skip2 = GatedSkipConnection(num_channels, exciter_hidden_ratio)
        self.skip3 = GatedSkipConnection(num_channels, exciter_hidden_ratio)
        self.skip4 = GatedSkipConnection(num_channels, exciter_hidden_ratio)

        self.qa_head = None

        self.checkpoint = lambda layer, x: layer.forward(x)

        self.num_channels = num_channels

    def initialize_weights(self) -> None:
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for layer in stage:
                layer.initialize_weights()

        for layer in [self.skip1, self.skip2, self.skip3, self.skip4]:
            layer.initialize_weights()

        if isinstance(self.qa_head, GlobalQualityAssessor):
            self.qa_head.initialize_weights()

    def add_qa_head(self, num_features: int) -> None:
        self.qa_head = GlobalQualityAssessor(self.num_channels, num_features)

    def remove_qa_head(self) -> None:
        self.qa_head = None

    def add_weight_norms(self) -> None:
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for layer in stage:
                layer.add_weight_norms()

        for layer in [self.skip1, self.skip2, self.skip3, self.skip4]:
            layer.add_weight_norms()

        if isinstance(self.qa_head, GlobalQualityAssessor):
            self.qa_head.add_weight_norms()

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        z1 = self.checkpoint(self.stage1, x)
        z1 = self.skip1.forward(x, z1)

        z2 = self.checkpoint(self.stage2, z1)
        z2 = self.skip2.forward(z1, z2)

        if isinstance(self.qa_head, GlobalQualityAssessor):
            z_qa = self.qa_head.forward(z2)
        else:
            z_qa = None

        z3 = self.checkpoint(self.stage3, z2)
        z3 = self.skip3.forward(z2, z3)

        z4 = self.checkpoint(self.stage4, z3)
        z4 = self.skip4.forward(z3, z4)

        return z4, z_qa


class UNet(Module):
    """
    A multiscale encoder/decoder network with residual connections.
    """

    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
        exciter_hidden_ratio: int,
    ):
        super().__init__()

        assert primary_layers > 1, "Number of primary layers must be greater than 1."

        assert (
            secondary_layers > 1
        ), "Number of secondary layers must be greater than 1."

        assert tertiary_layers > 1, "Number of tertiary layers must be greater than 1."

        assert (
            quaternary_layers > 1
        ), "Number of quaternary layers must be greater than 1."

        self.encoder = Encoder(
            primary_channels,
            ceil(primary_layers / 2),
            secondary_channels,
            ceil(secondary_layers / 2),
            tertiary_channels,
            ceil(tertiary_layers / 2),
            quaternary_channels,
            ceil(quaternary_layers / 2),
            hidden_ratio,
            exciter_hidden_ratio,
        )

        self.decoder = Decoder(
            quaternary_channels,
            floor(quaternary_layers / 2),
            tertiary_channels,
            floor(tertiary_layers / 2),
            secondary_channels,
            floor(secondary_layers / 2),
            primary_channels,
            floor(primary_layers / 2),
            hidden_ratio,
            exciter_hidden_ratio,
        )

        self.qa_head = None

        self.quaternary_channels = quaternary_channels

    def add_qa_head(self, num_features: int) -> None:
        self.qa_head = GlobalQualityAssessor(self.quaternary_channels, num_features)

    def remove_qa_head(self) -> None:
        self.qa_head = None

    def initialize_weights(self) -> None:
        self.encoder.initialize_weights()
        self.decoder.initialize_weights()

        if isinstance(self.qa_head, GlobalQualityAssessor):
            self.qa_head.initialize_weights()

    def freeze_weights(self) -> None:
        self.encoder.freeze_weights()
        self.decoder.freeze_weights()

    def unfreeze_weights(self) -> None:
        self.encoder.unfreeze_weights()
        self.decoder.unfreeze_weights()

    def add_weight_norms(self) -> None:
        self.encoder.add_weight_norms()
        self.decoder.add_weight_norms()

        if isinstance(self.qa_head, GlobalQualityAssessor):
            self.qa_head.add_weight_norms()

    def enable_activation_checkpointing(self) -> None:
        self.encoder.enable_activation_checkpointing()
        self.decoder.enable_activation_checkpointing()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z1, z2, z3, z4 = self.encoder.forward(x)

        if isinstance(self.qa_head, GlobalQualityAssessor):
            z_qa = self.qa_head.forward(z4)
        else:
            z_qa = None

        z = self.decoder.forward(z4, z3, z2, z1)

        return z, z_qa


class Encoder(Module):
    """An encoder subnetwork employing a deep stack of encoder blocks."""

    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
        exciter_hidden_ratio: int,
    ):
        super().__init__()

        assert primary_layers > 0, "Number of primary layers must be greater than 0."

        assert (
            secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."

        assert tertiary_layers > 0, "Number of tertiary layers must be greater than 0."

        assert (
            quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        new_encoder_block = partial(
            EncoderBlock,
            hidden_ratio=hidden_ratio,
            exciter_hidden_ratio=exciter_hidden_ratio,
        )

        self.stage1 = Sequential(
            *[new_encoder_block(primary_channels) for _ in range(primary_layers)]
        )

        self.stage2 = Sequential(
            *[new_encoder_block(secondary_channels) for _ in range(secondary_layers)]
        )

        self.stage3 = Sequential(
            *[new_encoder_block(tertiary_channels) for _ in range(tertiary_layers)]
        )

        self.stage4 = Sequential(
            *[new_encoder_block(quaternary_channels) for _ in range(quaternary_layers)]
        )

        self.downsample1 = PixelCrush(primary_channels, secondary_channels, 2)
        self.downsample2 = PixelCrush(secondary_channels, tertiary_channels, 2)
        self.downsample3 = PixelCrush(tertiary_channels, quaternary_channels, 2)

        self.checkpoint = lambda layer, x: layer.forward(x)

    def initialize_weights(self) -> None:
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for layer in stage:
                layer.initialize_weights()

        for layer in [self.downsample1, self.downsample2, self.downsample3]:
            layer.initialize_weights()

    def freeze_weights(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def add_weight_norms(self) -> None:
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for layer in stage:
                layer.add_weight_norms()

        for layer in [self.downsample1, self.downsample2, self.downsample3]:
            layer.add_weight_norms()

    def add_spectral_norms(self, num_iterations: int) -> None:
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for layer in stage:
                layer.add_spectral_norms(num_iterations)

        for layer in [self.downsample1, self.downsample2, self.downsample3]:
            layer.add_spectral_norms(num_iterations)

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1 = self.checkpoint(self.stage1, x)

        z2 = self.downsample1.forward(z1)

        z2 = self.checkpoint(self.stage2, z2)

        z3 = self.downsample2.forward(z2)

        z3 = self.checkpoint(self.stage3, z3)

        z4 = self.downsample3.forward(z3)

        z4 = self.checkpoint(self.stage4, z4)

        return z1, z2, z3, z4


class Decoder(Module):
    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
        exciter_hidden_ratio: int,
    ):
        super().__init__()

        assert primary_layers > 0, "Number of primary layers must be greater than 0."

        assert (
            secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."

        assert tertiary_layers > 0, "Number of tertiary layers must be greater than 0."

        assert (
            quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        new_decoder_block = partial(
            DecoderBlock,
            hidden_ratio=hidden_ratio,
            exciter_hidden_ratio=exciter_hidden_ratio,
        )

        self.stage1 = Sequential(
            *[new_decoder_block(primary_channels) for _ in range(primary_layers)]
        )

        self.stage2 = Sequential(
            *[new_decoder_block(secondary_channels) for _ in range(secondary_layers)]
        )

        self.stage3 = Sequential(
            *[new_decoder_block(tertiary_channels) for _ in range(tertiary_layers)]
        )

        self.stage4 = Sequential(
            *[new_decoder_block(quaternary_channels) for _ in range(quaternary_layers)]
        )

        self.upsample1 = SubpixelConv2d(primary_channels, secondary_channels, 2)
        self.upsample2 = SubpixelConv2d(secondary_channels, tertiary_channels, 2)
        self.upsample3 = SubpixelConv2d(tertiary_channels, quaternary_channels, 2)

        self.skip1 = GatedSkipConnection(secondary_channels, exciter_hidden_ratio)
        self.skip2 = GatedSkipConnection(tertiary_channels, exciter_hidden_ratio)
        self.skip3 = GatedSkipConnection(quaternary_channels, exciter_hidden_ratio)

        self.checkpoint = lambda layer, x: layer.forward(x)

    def initialize_weights(self) -> None:
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for layer in stage:
                layer.initialize_weights()

        for layer in [self.upsample1, self.upsample2, self.upsample3]:
            layer.initialize_weights()

        for layer in [self.skip1, self.skip2, self.skip3]:
            layer.initialize_weights()

    def freeze_weights(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def add_weight_norms(self) -> None:
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for layer in stage:
                layer.add_weight_norms()

        for layer in [self.upsample1, self.upsample2, self.upsample3]:
            layer.add_weight_norms()

        for layer in [self.skip1, self.skip2, self.skip3]:
            layer.add_weight_norms()

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    @staticmethod
    def crop_feature_maps(x: Tensor, size: FeatureMapSize) -> Tensor:
        """
        Center-crop or pad the feature maps to the target size. This is sometimes
        necessary due to rounding during down/upsampling operations.
        """

        _, _, h, w = x.shape

        target_h, target_w = size

        if h > target_h:
            start_h = (h - target_h) // 2
            end_h = start_h + target_h

            x = x[:, :, start_h:end_h, :]

        elif h < target_h:
            pad_h = target_h - h

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            x = pad(x, (0, 0, pad_top, pad_bottom))

        if w > target_w:
            start_w = (w - target_w) // 2
            end_w = start_w + target_w

            x = x[:, :, :, start_w:end_w]

        elif w < target_w:
            pad_w = target_w - w

            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            x = pad(x, (pad_left, pad_right, 0, 0))

        return x

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor, x4: Tensor) -> Tensor:
        z = self.checkpoint(self.stage1, x1)

        z = self.upsample1.forward(z)

        z = self.crop_feature_maps(z, x2.shape[2:4])

        z = self.skip1.forward(x2, z)

        z = self.checkpoint(self.stage2, z)

        z = self.upsample2.forward(z)

        z = self.crop_feature_maps(z, x3.shape[2:4])

        z = self.skip2.forward(x3, z)

        z = self.checkpoint(self.stage3, z)

        z = self.upsample3.forward(z)

        z = self.crop_feature_maps(z, x4.shape[2:4])

        z = self.skip3.forward(x4, z)

        z = self.checkpoint(self.stage4, z)

        return z


class EncoderBlock(Module):
    """A single encoder block consisting of a small convolutional network and a residual connection."""

    def __init__(self, num_channels: int, hidden_ratio: int, exciter_hidden_ratio: int):
        super().__init__()

        self.convnet = InvertedBottleneck(num_channels, hidden_ratio)

        self.skip = GatedSkipConnection(num_channels, exciter_hidden_ratio)

    def initialize_weights(self) -> None:
        self.convnet.initialize_weights()
        self.skip.initialize_weights()

    def add_weight_norms(self) -> None:
        self.convnet.add_weight_norms()
        self.skip.add_weight_norms()

    def add_spectral_norms(self, num_iterations: int) -> None:
        self.convnet.add_spectral_norms(num_iterations)
        self.skip.add_spectral_norms(num_iterations)

    def forward(self, x: Tensor) -> Tensor:
        z = self.convnet.forward(x)

        z = self.skip.forward(x, z)

        return z


class DecoderBlock(EncoderBlock):
    pass


class InvertedBottleneck(Module):
    """A two-layer convnet with wide non-linear activation pathway."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = Conv2d(
            num_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )

        self.conv2 = Conv2d(
            hidden_channels, num_channels, kernel_size=3, padding=1, bias=False
        )

        self.silu = SiLU()

    def initialize_weights(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def add_weight_norms(self) -> None:
        self.conv1 = weight_norm(self.conv1)
        self.conv2 = weight_norm(self.conv2)

    def add_spectral_norms(self, num_iterations: int) -> None:
        self.conv1 = spectral_norm(self.conv1, n_power_iterations=num_iterations)
        self.conv2 = spectral_norm(self.conv2, n_power_iterations=num_iterations)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1.forward(x)
        z = self.silu.forward(z)
        z = self.conv2.forward(z)

        return z


class GatedSkipConnection(Module):
    """
    A residual connection that selectively brings forward feature maps from the input based on
    the output of the current layer.
    """

    def __init__(self, num_channels: int, exciter_hidden_ratio: int):
        super().__init__()

        self.attention = ChannelAttention(num_channels, exciter_hidden_ratio)

    def initialize_weights(self) -> None:
        self.attention.initialize_weights()

    def add_weight_norms(self) -> None:
        self.attention.add_weight_norms()

    def add_spectral_norms(self, num_iterations: int) -> None:
        self.attention.add_spectral_norms(num_iterations)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        assert x.shape == z.shape, "Input and residual must have the same shape."

        x = self.attention.forward(x, z)

        return x + z


class ChannelAttention(Module):
    """A channel-wise attention mechanism for reweighting feature maps."""

    def __init__(self, num_channels: int, exciter_hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."

        self.pool = AdaptiveAvgPool2d(1)

        self.exciter = Exciter(num_channels, exciter_hidden_ratio)

        self.sigmoid = Sigmoid()

    def initialize_weights(self) -> None:
        self.exciter.initialize_weights()

    def add_weight_norms(self) -> None:
        self.exciter.add_weight_norms()

    def add_spectral_norms(self, num_iterations: int) -> None:
        self.exciter.add_spectral_norms(num_iterations)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        assert (
            x1.shape[:2] == x2.shape[:2]
        ), "X1 and X2 must have the same batch size and number of channels."

        z = self.pool.forward(x2)
        z = self.exciter.forward(z)
        z = self.sigmoid.forward(z)

        z = z.expand_as(x1)

        return z * x1


class Exciter(Module):
    """A small non-linear excitation network."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."

        assert hidden_ratio in {
            2,
            4,
            8,
            16,
        }, "Hidden ratio must be either 2, 4, 8, or 16."

        hidden_channels = num_channels // hidden_ratio

        self.conv1 = Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False)
        self.conv2 = Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False)

        self.silu = SiLU()

    def initialize_weights(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def add_weight_norms(self) -> None:
        self.conv1 = weight_norm(self.conv1)
        self.conv2 = weight_norm(self.conv2)

    def add_spectral_norms(self, num_iterations: int) -> None:
        self.conv1 = spectral_norm(self.conv1, n_power_iterations=num_iterations)
        self.conv2 = spectral_norm(self.conv2, n_power_iterations=num_iterations)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1.forward(x)
        z = self.silu.forward(z)
        z = self.conv2.forward(z)

        return z


class PixelCrush(Module):
    """Downsample the feature maps using strided convolution."""

    def __init__(self, in_channels: int, out_channels: int, crush_factor: int):
        super().__init__()

        assert in_channels > 0, "Input channels must be greater than 0."
        assert out_channels > 0, "Output channels must be greater than 0."

        assert crush_factor in {
            2,
            3,
            4,
        }, "Crush factor must be either 2, 3, or 4."

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=crush_factor,
            stride=crush_factor,
            bias=False,
        )

    def initialize_weights(self) -> None:
        self.conv.reset_parameters()

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def add_spectral_norms(self, num_iterations: int) -> None:
        self.conv = spectral_norm(self.conv, n_power_iterations=num_iterations)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class SubpixelConv2d(Module):
    """Upsample the feature maps using subpixel convolution."""

    def __init__(self, in_channels: int, out_channels: int, upscale_ratio: int):
        super().__init__()

        assert in_channels > 0, "Input channels must be greater than 0."
        assert out_channels > 0, "Output channels must be greater than 0."

        assert upscale_ratio in {
            2,
            3,
            4,
            8,
        }, "Upscale ratio must be either 2, 3, 4, or 8."

        out_channels = out_channels * upscale_ratio**2

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,  # Effective stride will be 1 / upscale_ratio.
            padding=1,
            bias=False,
        )

        self.shuffle = PixelShuffle(upscale_ratio)

    def initialize_weights(self) -> None:
        self.conv.reset_parameters()

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def add_spectral_norms(self, num_iterations: int) -> None:
        self.conv = spectral_norm(self.conv, n_power_iterations=num_iterations)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)
        z = self.shuffle.forward(z)

        return z


class GlobalQualityAssessor(Module):
    """A multi-regression head for estimating the total amount of degradation present in the input image."""

    def __init__(self, num_channels: int, num_features: int):
        super().__init__()

        assert (
            num_features > 0
        ), "Number of degradation features must be greater than 0."

        self.pool = AdaptiveAvgPool2d(1)

        self.conv = Conv2d(num_channels, num_features, kernel_size=1)

        self.flatten = Flatten(start_dim=1)

    def initialize_weights(self) -> None:
        self.conv.reset_parameters()

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        z = self.pool.forward(x)
        z = self.conv.forward(z)
        z = self.flatten.forward(z)

        return z


class SuperResolver(Module):
    """A decoder head for upscaling the input feature maps beyond their original size."""

    def __init__(self, in_channels: int, upscale_ratio: int):
        super().__init__()

        assert upscale_ratio in {
            2,
            3,
            4,
            8,
        }, "Upscale ratio must be either 2, 3, 4, or 8."

        self.upscale = SubpixelConv2d(in_channels, 3, upscale_ratio)

    def initialize_weights(self) -> None:
        self.upscale.initialize_weights()

    def add_weight_norms(self) -> None:
        self.upscale.add_weight_norms()

    def forward(self, x: Tensor) -> Tensor:
        z = self.upscale.forward(x)

        return z


class Bouncer(Critic, Module):
    """A discriminator network for detecting real and fake images for adversarial training."""

    AVAILABLE_MODEL_SIZES = {"small", "medium", "large"}

    @classmethod
    def from_preconfigured(cls, model_size: str) -> Self:
        """Return a new pre-configured model."""

        match model_size:
            case "small":
                primary_channels = 48
                primary_layers = 1
                secondary_channels = 96
                secondary_layers = 1
                tertiary_channels = 192
                tertiary_layers = 1
                quaternary_channels = 384
                quaternary_layers = 1

            case "medium":
                primary_channels = 64
                primary_layers = 2
                secondary_channels = 128
                secondary_layers = 2
                tertiary_channels = 256
                tertiary_layers = 2
                quaternary_channels = 512
                quaternary_layers = 2

            case "large":
                primary_channels = 96
                primary_layers = 3
                secondary_channels = 192
                secondary_layers = 3
                tertiary_channels = 384
                tertiary_layers = 3
                quaternary_channels = 768
                quaternary_layers = 3

            case _:
                raise ValueError("Invalid model size.")

        hidden_ratio = 2
        exciter_hidden_ratio = 8

        return cls(
            primary_channels,
            primary_layers,
            secondary_channels,
            secondary_layers,
            tertiary_channels,
            tertiary_layers,
            quaternary_channels,
            quaternary_layers,
            hidden_ratio,
            exciter_hidden_ratio,
        )

    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
        exciter_hidden_ratio: int,
    ):
        super().__init__()

        self.stem = FanOutProjection(3, primary_channels)

        self.body = Encoder(
            primary_channels,
            primary_layers,
            secondary_channels,
            secondary_layers,
            tertiary_channels,
            tertiary_layers,
            quaternary_channels,
            quaternary_layers,
            hidden_ratio,
            exciter_hidden_ratio,
        )

        self.head = PatchDiscriminator(quaternary_channels)

        self.sigmoid = Sigmoid()

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def add_spectral_norms(self, num_iterations: int) -> None:
        """Add spectral normalization to the network."""

        self.stem.add_spectral_norms(num_iterations)
        self.body.add_spectral_norms(num_iterations)
        self.head.add_spectral_norms(num_iterations)

    def remove_parameterizations(self) -> None:
        """Remove all parameterizations."""

        for module in self.modules():
            if is_parametrized(module):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def enable_activation_checkpointing(self) -> None:
        """Enable activation checkpointing for the detector."""

        self.body.enable_activation_checkpointing()

    def forward(self, x: Tensor) -> Tensor:
        z = self.stem.forward(x)

        _, _, _, z4 = self.body.forward(z)

        z = self.head.forward(z4)

        return z

    @torch.inference_mode()
    def predict(self, x: Tensor) -> Tensor:
        """Convenience method for inference."""

        z = self.forward(x)
        z = self.sigmoid.forward(z)

        return z


class PatchDiscriminator(Module):
    """
    A patch-based discriminator head for estimating the probability that local patches of the
    input feature maps are real or fake.
    """

    def __init__(self, num_channels: int):
        super().__init__()

        self.downsample = PixelCrush(num_channels, num_channels, 2)

        self.conv = Conv2d(num_channels, 1, kernel_size=1, bias=False)

    def add_spectral_norms(self, num_iterations: int) -> None:
        self.downsample.add_spectral_norms(num_iterations)

        self.conv = spectral_norm(self.conv, n_power_iterations=num_iterations)

    def forward(self, x: Tensor) -> Tensor:
        z = self.downsample.forward(x)
        z = self.conv.forward(z)

        return z
