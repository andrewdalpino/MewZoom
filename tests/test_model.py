import unittest

import torch
from torch import Tensor

from src.mewzoom.model import (
    ChannelAttention,
    Decoder,
    Encoder,
    EncoderBlock,
    Exciter,
    FanOutProjection,
    GatedSkipConnection,
    GlobalQualityAssessor,
    InvertedBottleneck,
    MewZoom,
    MewZoomTrunkNet,
    MewZoomUNet,
    ONNXModel,
    PatchDiscriminator,
    PixelCrush,
    SubpixelConv2d,
    SuperResolver,
    TrunkNet,
    UNet,
    Bouncer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rand(batch: int, channels: int, h: int, w: int) -> Tensor:
    return torch.rand(batch, channels, h, w)


# Minimal config for MewZoomTrunkNet that satisfies all constraints
TRUNK_KWARGS = dict(
    upscale_ratio=2,
    num_channels=16,
    num_layers=4,
    hidden_ratio=1,
    exciter_hidden_ratio=2,
)

# Minimal config for MewZoomUNet that satisfies all constraints
UNET_KWARGS = dict(
    upscale_ratio=2,
    primary_channels=8,
    primary_layers=2,
    secondary_channels=16,
    secondary_layers=2,
    tertiary_channels=32,
    tertiary_layers=2,
    quaternary_channels=64,
    quaternary_layers=2,
    hidden_ratio=1,
    exciter_hidden_ratio=2,
)


# ---------------------------------------------------------------------------
# FanOutProjection
# ---------------------------------------------------------------------------

class TestFanOutProjection(unittest.TestCase):

    def test_forward_expands_channels(self):
        layer = FanOutProjection(3, 16)
        x = _rand(2, 3, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, (2, 16, 8, 8))

    def test_invalid_in_channels_raises(self):
        with self.assertRaises(AssertionError):
            FanOutProjection(0, 16)

    def test_out_channels_must_exceed_in_channels(self):
        with self.assertRaises(AssertionError):
            FanOutProjection(16, 8)

    def test_initialize_weights(self):
        layer = FanOutProjection(3, 16)
        layer.initialize_weights()

    def test_add_weight_norms(self):
        layer = FanOutProjection(3, 16)
        layer.add_weight_norms()
        x = _rand(1, 3, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, (1, 16, 8, 8))


# ---------------------------------------------------------------------------
# InvertedBottleneck
# ---------------------------------------------------------------------------

class TestInvertedBottleneck(unittest.TestCase):

    def test_forward_preserves_shape(self):
        layer = InvertedBottleneck(16, hidden_ratio=2)
        x = _rand(2, 16, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_invalid_hidden_ratio_raises(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(16, hidden_ratio=3)

    def test_invalid_num_channels_raises(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(0, hidden_ratio=2)

    def test_initialize_weights(self):
        layer = InvertedBottleneck(16, hidden_ratio=1)
        layer.initialize_weights()

    def test_add_weight_norms(self):
        layer = InvertedBottleneck(16, hidden_ratio=2)
        layer.add_weight_norms()
        out = layer(_rand(1, 16, 8, 8))
        self.assertEqual(out.shape, (1, 16, 8, 8))


# ---------------------------------------------------------------------------
# Exciter
# ---------------------------------------------------------------------------

class TestExciter(unittest.TestCase):

    def test_forward_preserves_shape(self):
        layer = Exciter(16, hidden_ratio=2)
        x = _rand(2, 16, 1, 1)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_invalid_hidden_ratio_raises(self):
        with self.assertRaises(AssertionError):
            Exciter(16, hidden_ratio=3)

    def test_invalid_num_channels_raises(self):
        with self.assertRaises(AssertionError):
            Exciter(0, hidden_ratio=2)

    def test_all_valid_hidden_ratios(self):
        for ratio in (2, 4, 8, 16):
            layer = Exciter(32, hidden_ratio=ratio)
            out = layer(_rand(1, 32, 1, 1))
            self.assertEqual(out.shape, (1, 32, 1, 1))


# ---------------------------------------------------------------------------
# ChannelAttention
# ---------------------------------------------------------------------------

class TestChannelAttention(unittest.TestCase):

    def test_forward_preserves_spatial_shape_of_x1(self):
        layer = ChannelAttention(16, exciter_hidden_ratio=2)
        x1 = _rand(2, 16, 8, 8)
        x2 = _rand(2, 16, 4, 4)
        out = layer(x1, x2)
        self.assertEqual(out.shape, x1.shape)

    def test_mismatched_channels_raises(self):
        layer = ChannelAttention(16, exciter_hidden_ratio=2)
        x1 = _rand(1, 16, 8, 8)
        x2 = _rand(1, 8, 8, 8)
        with self.assertRaises(AssertionError):
            layer(x1, x2)

    def test_add_weight_norms(self):
        layer = ChannelAttention(16, exciter_hidden_ratio=2)
        layer.add_weight_norms()


# ---------------------------------------------------------------------------
# GatedSkipConnection
# ---------------------------------------------------------------------------

class TestGatedSkipConnection(unittest.TestCase):

    def test_forward_preserves_shape(self):
        layer = GatedSkipConnection(16, exciter_hidden_ratio=2)
        x = _rand(2, 16, 8, 8)
        z = _rand(2, 16, 8, 8)
        out = layer(x, z)
        self.assertEqual(out.shape, x.shape)

    def test_mismatched_shapes_raises(self):
        layer = GatedSkipConnection(16, exciter_hidden_ratio=2)
        x = _rand(1, 16, 8, 8)
        z = _rand(1, 16, 4, 4)
        with self.assertRaises(AssertionError):
            layer(x, z)

    def test_initialize_weights(self):
        layer = GatedSkipConnection(16, exciter_hidden_ratio=2)
        layer.initialize_weights()


# ---------------------------------------------------------------------------
# PixelCrush
# ---------------------------------------------------------------------------

class TestPixelCrush(unittest.TestCase):

    def test_forward_halves_spatial_dims(self):
        layer = PixelCrush(16, 32, crush_factor=2)
        x = _rand(2, 16, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, (2, 32, 4, 4))

    def test_invalid_crush_factor_raises(self):
        with self.assertRaises(AssertionError):
            PixelCrush(16, 32, crush_factor=5)

    def test_invalid_in_channels_raises(self):
        with self.assertRaises(AssertionError):
            PixelCrush(0, 32, crush_factor=2)

    def test_invalid_out_channels_raises(self):
        with self.assertRaises(AssertionError):
            PixelCrush(16, 0, crush_factor=2)

    def test_all_valid_crush_factors(self):
        for factor in (2, 3, 4):
            layer = PixelCrush(16, 32, crush_factor=factor)
            x = _rand(1, 16, 12, 12)
            out = layer(x)
            self.assertEqual(out.shape[1], 32)
            self.assertEqual(out.shape[2], 12 // factor)

    def test_add_weight_norms(self):
        layer = PixelCrush(16, 32, crush_factor=2)
        layer.add_weight_norms()


# ---------------------------------------------------------------------------
# SubpixelConv2d
# ---------------------------------------------------------------------------

class TestSubpixelConv2d(unittest.TestCase):

    def test_forward_doubles_spatial_dims(self):
        layer = SubpixelConv2d(16, 4, upscale_ratio=2)
        x = _rand(2, 16, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, 16, 16))

    def test_invalid_upscale_ratio_raises(self):
        with self.assertRaises(AssertionError):
            SubpixelConv2d(16, 32, upscale_ratio=5)

    def test_invalid_in_channels_raises(self):
        with self.assertRaises(AssertionError):
            SubpixelConv2d(0, 32, upscale_ratio=2)

    def test_invalid_out_channels_raises(self):
        with self.assertRaises(AssertionError):
            SubpixelConv2d(16, 0, upscale_ratio=2)

    def test_initialize_weights(self):
        layer = SubpixelConv2d(16, 4, upscale_ratio=2)
        layer.initialize_weights()

    def test_add_weight_norms(self):
        layer = SubpixelConv2d(16, 4, upscale_ratio=2)
        layer.add_weight_norms()


# ---------------------------------------------------------------------------
# GlobalQualityAssessor
# ---------------------------------------------------------------------------

class TestGlobalQualityAssessor(unittest.TestCase):

    def test_forward_output_shape(self):
        layer = GlobalQualityAssessor(num_channels=16, num_features=4)
        x = _rand(2, 16, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, (2, 4))

    def test_invalid_num_features_raises(self):
        with self.assertRaises(AssertionError):
            GlobalQualityAssessor(num_channels=16, num_features=0)

    def test_initialize_weights(self):
        layer = GlobalQualityAssessor(16, 4)
        layer.initialize_weights()

    def test_add_weight_norms(self):
        layer = GlobalQualityAssessor(16, 4)
        layer.add_weight_norms()
        out = layer(_rand(1, 16, 8, 8))
        self.assertEqual(out.shape, (1, 4))


# ---------------------------------------------------------------------------
# SuperResolver
# ---------------------------------------------------------------------------

class TestSuperResolver(unittest.TestCase):

    def test_forward_upscales_2x(self):
        layer = SuperResolver(in_channels=16, upscale_ratio=2)
        x = _rand(1, 16, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, (1, 3, 16, 16))

    def test_forward_upscales_4x(self):
        layer = SuperResolver(in_channels=16, upscale_ratio=4)
        x = _rand(1, 16, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_invalid_upscale_ratio_raises(self):
        with self.assertRaises(AssertionError):
            SuperResolver(16, upscale_ratio=5)

    def test_initialize_weights(self):
        layer = SuperResolver(16, upscale_ratio=2)
        layer.initialize_weights()


# ---------------------------------------------------------------------------
# EncoderBlock / DecoderBlock
# ---------------------------------------------------------------------------

class TestEncoderBlock(unittest.TestCase):

    def test_forward_preserves_shape(self):
        block = EncoderBlock(num_channels=16, hidden_ratio=2, exciter_hidden_ratio=2)
        x = _rand(2, 16, 8, 8)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_initialize_weights(self):
        block = EncoderBlock(16, hidden_ratio=1, exciter_hidden_ratio=4)
        block.initialize_weights()

    def test_add_weight_norms(self):
        block = EncoderBlock(16, hidden_ratio=2, exciter_hidden_ratio=2)
        block.add_weight_norms()
        out = block(_rand(1, 16, 8, 8))
        self.assertEqual(out.shape, (1, 16, 8, 8))


# ---------------------------------------------------------------------------
# TrunkNet
# ---------------------------------------------------------------------------

class TestTrunkNet(unittest.TestCase):

    def _make(self, **overrides):
        kwargs = dict(num_channels=16, num_layers=4, hidden_ratio=1, exciter_hidden_ratio=2)
        kwargs.update(overrides)
        return TrunkNet(**kwargs)

    def test_forward_returns_tensor_and_none_without_qa_head(self):
        net = self._make()
        x = _rand(1, 16, 8, 8)
        out, qa = net(x)
        self.assertIsInstance(out, Tensor)
        self.assertIsNone(qa)
        self.assertEqual(out.shape, x.shape)

    def test_forward_returns_qa_tensor_with_qa_head(self):
        net = self._make()
        net.add_qa_head(num_features=4)
        x = _rand(1, 16, 8, 8)
        out, qa = net(x)
        self.assertIsInstance(out, Tensor)
        self.assertIsInstance(qa, Tensor)
        self.assertEqual(qa.shape, (1, 4))

    def test_add_and_remove_qa_head(self):
        net = self._make()
        net.add_qa_head(num_features=3)
        self.assertIsNotNone(net.qa_head)
        net.remove_qa_head()
        self.assertIsNone(net.qa_head)

    def test_invalid_num_layers_raises(self):
        with self.assertRaises(AssertionError):
            self._make(num_layers=3)

    def test_enable_activation_checkpointing(self):
        net = self._make()
        net.enable_activation_checkpointing()
        x = _rand(1, 16, 8, 8)
        out, _ = net(x)
        self.assertEqual(out.shape, x.shape)

    def test_initialize_weights(self):
        net = self._make()
        net.initialize_weights()

    def test_add_weight_norms(self):
        net = self._make()
        net.add_weight_norms()
        out, _ = net(_rand(1, 16, 8, 8))
        self.assertEqual(out.shape, (1, 16, 8, 8))


# ---------------------------------------------------------------------------
# MewZoomTrunkNet
# ---------------------------------------------------------------------------

class TestMewZoomTrunkNet(unittest.TestCase):

    def setUp(self):
        self.model = MewZoomTrunkNet(**TRUNK_KWARGS)

    def test_forward_output_shape(self):
        x = _rand(1, 3, 16, 16)
        out, qa = self.model(x)
        self.assertEqual(out.shape, (1, 3, 32, 32))
        self.assertIsNone(qa)

    def test_upscale_output_shape(self):
        x = _rand(1, 3, 16, 16)
        out = self.model.upscale(x)
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_upscale_output_clamped(self):
        x = _rand(1, 3, 16, 16)
        out = self.model.upscale(x)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_num_params_positive(self):
        self.assertGreater(self.model.num_params, 0)

    def test_num_trainable_params_equals_num_params_by_default(self):
        self.assertEqual(self.model.num_params, self.model.num_trainable_params)

    def test_invalid_upscale_ratio_raises(self):
        with self.assertRaises(AssertionError):
            MewZoomTrunkNet(upscale_ratio=5, num_channels=16, num_layers=4,
                            hidden_ratio=1, exciter_hidden_ratio=2)

    def test_add_qa_head_and_forward(self):
        self.model.add_qa_head(num_features=3)
        x = _rand(1, 3, 16, 16)
        out, qa = self.model(x)
        self.assertEqual(qa.shape, (1, 3))

    def test_remove_qa_head(self):
        self.model.add_qa_head(num_features=3)
        self.model.remove_qa_head()
        _, qa = self.model(_rand(1, 3, 16, 16))
        self.assertIsNone(qa)

    def test_initialize_weights(self):
        self.model.initialize_weights()

    def test_add_weight_norms(self):
        self.model.add_weight_norms()
        out, _ = self.model(_rand(1, 3, 16, 16))
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_enable_activation_checkpointing(self):
        self.model.enable_activation_checkpointing()
        out, _ = self.model(_rand(1, 3, 16, 16))
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_all_valid_upscale_ratios(self):
        for ratio in MewZoomTrunkNet.AVAILABLE_UPSCALE_RATIOS:
            model = MewZoomTrunkNet(
                upscale_ratio=ratio, num_channels=16, num_layers=4,
                hidden_ratio=1, exciter_hidden_ratio=2
            )
            x = _rand(1, 3, 8, 8)
            out, _ = model(x)
            self.assertEqual(out.shape, (1, 3, 8 * ratio, 8 * ratio))


# ---------------------------------------------------------------------------
# MewZoomUNet
# ---------------------------------------------------------------------------

class TestMewZoomUNet(unittest.TestCase):

    def setUp(self):
        self.model = MewZoomUNet(**UNET_KWARGS)

    def test_forward_output_shape(self):
        x = _rand(1, 3, 32, 32)
        out, qa = self.model(x)
        self.assertEqual(out.shape, (1, 3, 64, 64))
        self.assertIsNone(qa)

    def test_upscale_output_clamped(self):
        x = _rand(1, 3, 32, 32)
        out = self.model.upscale(x)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_num_params_positive(self):
        self.assertGreater(self.model.num_params, 0)

    def test_invalid_upscale_ratio_raises(self):
        kwargs = dict(UNET_KWARGS)
        kwargs["upscale_ratio"] = 5
        with self.assertRaises(AssertionError):
            MewZoomUNet(**kwargs)

    def test_add_and_remove_qa_head(self):
        self.model.add_qa_head(num_features=3)
        _, qa = self.model(_rand(1, 3, 32, 32))
        self.assertIsNotNone(qa)
        self.model.remove_qa_head()
        _, qa = self.model(_rand(1, 3, 32, 32))
        self.assertIsNone(qa)

    def test_initialize_weights(self):
        self.model.initialize_weights()

    def test_add_weight_norms(self):
        self.model.add_weight_norms()
        out, _ = self.model(_rand(1, 3, 32, 32))
        self.assertEqual(out.shape, (1, 3, 64, 64))

    def test_enable_activation_checkpointing(self):
        self.model.enable_activation_checkpointing()
        out, _ = self.model(_rand(1, 3, 32, 32))
        self.assertEqual(out.shape, (1, 3, 64, 64))


# ---------------------------------------------------------------------------
# MewZoom (factory)
# ---------------------------------------------------------------------------

class TestMewZoom(unittest.TestCase):

    def test_trunknet_architecture(self):
        model = MewZoom(architecture="trunknet", **TRUNK_KWARGS)
        out, _ = model(_rand(1, 3, 16, 16))
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_unet_architecture(self):
        model = MewZoom(architecture="unet", **UNET_KWARGS)
        out, _ = model(_rand(1, 3, 32, 32))
        self.assertEqual(out.shape, (1, 3, 64, 64))

    def test_invalid_architecture_raises(self):
        with self.assertRaises(ValueError):
            MewZoom(architecture="resnet", **TRUNK_KWARGS)

    def test_architecture_name_case_insensitive(self):
        model = MewZoom(architecture="TrunkNet", **TRUNK_KWARGS)
        out, _ = model(_rand(1, 3, 16, 16))
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_upscale_output_clamped(self):
        model = MewZoom(architecture="trunknet", **TRUNK_KWARGS)
        out = model.upscale(_rand(1, 3, 16, 16))
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_initialize_weights(self):
        model = MewZoom(architecture="trunknet", **TRUNK_KWARGS)
        model.initialize_weights()

    def test_remove_parameterizations(self):
        model = MewZoom(architecture="trunknet", **TRUNK_KWARGS)
        model.model.add_weight_norms()
        model.remove_parameterizations()
        # Should complete without error and forward still works
        out, _ = model(_rand(1, 3, 16, 16))
        self.assertEqual(out.shape, (1, 3, 32, 32))


# ---------------------------------------------------------------------------
# ONNXModel
# ---------------------------------------------------------------------------

class TestONNXModel(unittest.TestCase):

    def test_forward_delegates_to_upscale(self):
        inner = MewZoomTrunkNet(**TRUNK_KWARGS)
        wrapper = ONNXModel(inner)
        x = _rand(1, 3, 16, 16)
        out = wrapper(x)
        self.assertEqual(out.shape, (1, 3, 32, 32))
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TestEncoder(unittest.TestCase):

    def _make(self, **overrides):
        kwargs = dict(
            primary_channels=8,
            primary_layers=1,
            secondary_channels=16,
            secondary_layers=1,
            tertiary_channels=32,
            tertiary_layers=1,
            quaternary_channels=64,
            quaternary_layers=1,
            hidden_ratio=1,
            exciter_hidden_ratio=2,
        )
        kwargs.update(overrides)
        return Encoder(**kwargs)

    def test_forward_returns_four_feature_maps(self):
        enc = self._make()
        x = _rand(1, 8, 32, 32)
        z1, z2, z3, z4 = enc(x)
        self.assertEqual(z1.shape[1], 8)
        self.assertEqual(z2.shape[1], 16)
        self.assertEqual(z3.shape[1], 32)
        self.assertEqual(z4.shape[1], 64)

    def test_spatial_dims_decrease_with_depth(self):
        enc = self._make()
        x = _rand(1, 8, 32, 32)
        z1, z2, z3, z4 = enc(x)
        self.assertGreater(z1.shape[2], z2.shape[2])
        self.assertGreater(z2.shape[2], z3.shape[2])
        self.assertGreater(z3.shape[2], z4.shape[2])

    def test_invalid_primary_layers_raises(self):
        with self.assertRaises(AssertionError):
            self._make(primary_layers=0)

    def test_invalid_secondary_layers_raises(self):
        with self.assertRaises(AssertionError):
            self._make(secondary_layers=0)

    def test_invalid_tertiary_layers_raises(self):
        with self.assertRaises(AssertionError):
            self._make(tertiary_layers=0)

    def test_invalid_quaternary_layers_raises(self):
        with self.assertRaises(AssertionError):
            self._make(quaternary_layers=0)

    def test_initialize_weights(self):
        enc = self._make()
        enc.initialize_weights()

    def test_freeze_and_unfreeze_weights(self):
        enc = self._make()
        enc.freeze_weights()
        for param in enc.parameters():
            self.assertFalse(param.requires_grad)
        enc.unfreeze_weights()
        for param in enc.parameters():
            self.assertTrue(param.requires_grad)

    def test_add_weight_norms(self):
        enc = self._make()
        enc.add_weight_norms()
        x = _rand(1, 8, 32, 32)
        z1, z2, z3, z4 = enc(x)
        self.assertEqual(z1.shape[1], 8)

    def test_enable_activation_checkpointing(self):
        enc = self._make()
        enc.enable_activation_checkpointing()
        x = _rand(1, 8, 32, 32)
        z1, z2, z3, z4 = enc(x)
        self.assertEqual(z4.shape[1], 64)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class TestUNet(unittest.TestCase):

    def _make(self, **overrides):
        kwargs = dict(
            primary_channels=8,
            primary_layers=2,
            secondary_channels=16,
            secondary_layers=2,
            tertiary_channels=32,
            tertiary_layers=2,
            quaternary_channels=64,
            quaternary_layers=2,
            hidden_ratio=1,
            exciter_hidden_ratio=2,
        )
        kwargs.update(overrides)
        return UNet(**kwargs)

    def test_forward_returns_tensor_and_none_without_qa_head(self):
        net = self._make()
        x = _rand(1, 8, 32, 32)
        out, qa = net(x)
        self.assertIsInstance(out, Tensor)
        self.assertIsNone(qa)
        self.assertEqual(out.shape[1], 8)

    def test_forward_output_preserves_spatial_dims(self):
        net = self._make()
        x = _rand(1, 8, 32, 32)
        out, _ = net(x)
        self.assertEqual(out.shape[2], x.shape[2])
        self.assertEqual(out.shape[3], x.shape[3])

    def test_forward_returns_qa_tensor_with_qa_head(self):
        net = self._make()
        net.add_qa_head(num_features=5)
        x = _rand(1, 8, 32, 32)
        out, qa = net(x)
        self.assertIsInstance(qa, Tensor)
        self.assertEqual(qa.shape, (1, 5))

    def test_add_and_remove_qa_head(self):
        net = self._make()
        net.add_qa_head(num_features=3)
        self.assertIsNotNone(net.qa_head)
        net.remove_qa_head()
        self.assertIsNone(net.qa_head)

    def test_invalid_primary_layers_raises(self):
        with self.assertRaises(AssertionError):
            self._make(primary_layers=1)

    def test_invalid_secondary_layers_raises(self):
        with self.assertRaises(AssertionError):
            self._make(secondary_layers=1)

    def test_invalid_tertiary_layers_raises(self):
        with self.assertRaises(AssertionError):
            self._make(tertiary_layers=1)

    def test_invalid_quaternary_layers_raises(self):
        with self.assertRaises(AssertionError):
            self._make(quaternary_layers=1)

    def test_initialize_weights(self):
        net = self._make()
        net.initialize_weights()

    def test_freeze_and_unfreeze_weights(self):
        net = self._make()
        net.freeze_weights()
        for param in net.encoder.parameters():
            self.assertFalse(param.requires_grad)
        net.unfreeze_weights()
        for param in net.encoder.parameters():
            self.assertTrue(param.requires_grad)

    def test_add_weight_norms(self):
        net = self._make()
        net.add_weight_norms()
        out, _ = net(_rand(1, 8, 32, 32))
        self.assertEqual(out.shape[1], 8)

    def test_enable_activation_checkpointing(self):
        net = self._make()
        net.enable_activation_checkpointing()
        out, _ = net(_rand(1, 8, 32, 32))
        self.assertEqual(out.shape[1], 8)


# ---------------------------------------------------------------------------
# Decoder.crop_feature_maps (static method)
# ---------------------------------------------------------------------------

class TestDecoderCropFeatureMaps(unittest.TestCase):

    def test_no_op_when_sizes_match(self):
        x = _rand(1, 8, 16, 16)
        result = Decoder.crop_feature_maps(x, (16, 16))
        self.assertEqual(result.shape, x.shape)

    def test_crops_height_when_too_tall(self):
        x = _rand(1, 8, 18, 16)
        result = Decoder.crop_feature_maps(x, (16, 16))
        self.assertEqual(result.shape, (1, 8, 16, 16))

    def test_crops_width_when_too_wide(self):
        x = _rand(1, 8, 16, 18)
        result = Decoder.crop_feature_maps(x, (16, 16))
        self.assertEqual(result.shape, (1, 8, 16, 16))

    def test_pads_height_when_too_short(self):
        x = _rand(1, 8, 14, 16)
        result = Decoder.crop_feature_maps(x, (16, 16))
        self.assertEqual(result.shape, (1, 8, 16, 16))

    def test_pads_width_when_too_narrow(self):
        x = _rand(1, 8, 16, 14)
        result = Decoder.crop_feature_maps(x, (16, 16))
        self.assertEqual(result.shape, (1, 8, 16, 16))

    def test_crop_and_pad_combined(self):
        # Taller but narrower
        x = _rand(1, 8, 18, 14)
        result = Decoder.crop_feature_maps(x, (16, 16))
        self.assertEqual(result.shape, (1, 8, 16, 16))


# ---------------------------------------------------------------------------
# PatchDiscriminator
# ---------------------------------------------------------------------------

class TestPatchDiscriminator(unittest.TestCase):

    def test_forward_output_shape(self):
        layer = PatchDiscriminator(num_channels=64)
        x = _rand(1, 64, 16, 16)
        out = layer(x)
        # PixelCrush halves spatial dims, then Conv2d(kernel=1) keeps them
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 1)

    def test_add_spectral_norms(self):
        layer = PatchDiscriminator(num_channels=64)
        layer.add_spectral_norms(num_iterations=1)
        out = layer(_rand(1, 64, 16, 16))
        self.assertEqual(out.shape[1], 1)


# ---------------------------------------------------------------------------
# Bouncer
# ---------------------------------------------------------------------------

class TestBouncer(unittest.TestCase):

    def _small_bouncer(self):
        return Bouncer.from_preconfigured("small")

    def test_from_preconfigured_small(self):
        model = Bouncer.from_preconfigured("small")
        self.assertIsInstance(model, Bouncer)

    def test_from_preconfigured_medium(self):
        model = Bouncer.from_preconfigured("medium")
        self.assertIsInstance(model, Bouncer)

    def test_from_preconfigured_large(self):
        model = Bouncer.from_preconfigured("large")
        self.assertIsInstance(model, Bouncer)

    def test_from_preconfigured_invalid_raises(self):
        with self.assertRaises(ValueError):
            Bouncer.from_preconfigured("xlarge")

    def test_forward_output_shape(self):
        model = self._small_bouncer()
        x = _rand(1, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 1)

    def test_predict_output_in_range(self):
        model = self._small_bouncer()
        x = _rand(1, 3, 32, 32)
        out = model.predict(x)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_num_trainable_params_positive(self):
        model = self._small_bouncer()
        self.assertGreater(model.num_trainable_params, 0)

    def test_add_spectral_norms_and_forward(self):
        model = self._small_bouncer()
        model.add_spectral_norms(num_iterations=1)
        x = _rand(1, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape[0], 1)

    def test_remove_parameterizations(self):
        model = self._small_bouncer()
        model.add_spectral_norms(num_iterations=1)
        model.remove_parameterizations()
        out = model(_rand(1, 3, 32, 32))
        self.assertEqual(out.shape[0], 1)

    def test_enable_activation_checkpointing(self):
        model = self._small_bouncer()
        model.enable_activation_checkpointing()
        out = model(_rand(1, 3, 32, 32))
        self.assertEqual(out.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
