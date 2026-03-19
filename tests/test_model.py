import unittest

import torch
from torch import Tensor
from torch.nn import Conv2d

from src.mewzoom.model import (
    MewZoom,
    ONNXModel,
    FanOutProjection,
    UNet,
    Encoder,
    EncoderBlock,
    Decoder,
    DecoderBlock,
    InvertedBottleneck,
    ResidualConnection,
    PixelCrush,
    SubpixelConv2d,
    QualityAssessor,
    SuperResolver,
    ChannelLoRA,
    Bouncer,
    FeatureDetector,
    DetectorBlock,
    DepthwiseSeparableConv2d,
    PatchDiscriminator,
)


class TestMewZoom(unittest.TestCase):
    def test_init_valid_upscale_ratio(self):
        model = MewZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        self.assertEqual(model.upscale_ratio, 2)

    def test_init_invalid_upscale_ratio(self):
        with self.assertRaises(AssertionError):
            MewZoom(
                upscale_ratio=5,
                primary_channels=32,
                primary_layers=2,
                secondary_channels=64,
                secondary_layers=2,
                tertiary_channels=128,
                tertiary_layers=2,
                quaternary_channels=256,
                quaternary_layers=2,
                hidden_ratio=4,
            )

    def test_forward_output_shape(self):
        model = MewZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        x = torch.randn(1, 3, 32, 32)
        z, z_qa = model.forward(x)
        self.assertEqual(z.shape, (1, 3, 64, 64))
        self.assertEqual(z_qa.shape[0], 1)

    def test_num_params(self):
        model = MewZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        self.assertGreater(model.num_params, 0)

    def test_num_trainable_params(self):
        model = MewZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        self.assertEqual(model.num_params, model.num_trainable_params)

    def test_freeze_parameters(self):
        model = MewZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        model.freeze_parameters()
        for param in model.parameters():
            self.assertFalse(param.requires_grad)

    def test_add_qa_head(self):
        model = MewZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        model.add_qa_head(num_features=5)
        x = torch.randn(1, 3, 32, 32)
        z, z_qa = model.forward(x)
        self.assertEqual(z_qa.shape, (1, 5))

    def test_add_lora_adapters(self):
        model = MewZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        model.add_lora_adapters(rank=4, alpha=1.0)
        x = torch.randn(1, 3, 32, 32)
        z, _ = model.forward(x)
        self.assertEqual(z.shape, (1, 3, 64, 64))

    def test_upscale(self):
        model = MewZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        z = model.upscale(x)
        self.assertEqual(z.shape, (1, 3, 64, 64))
        self.assertTrue(torch.all(z >= 0) and torch.all(z <= 1))


class TestONNXModel(unittest.TestCase):
    def test_forward_output_shape(self):
        model = MewZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        model.eval()
        onnx_model = ONNXModel(model)
        x = torch.randn(1, 3, 32, 32)
        z = onnx_model.forward(x)
        self.assertEqual(z.shape, (1, 3, 64, 64))


class TestFanOutProjection(unittest.TestCase):
    def test_init_valid_channels(self):
        proj = FanOutProjection(in_channels=3, out_channels=32)
        self.assertIsNotNone(proj.conv)

    def test_init_invalid_in_channels(self):
        with self.assertRaises(AssertionError):
            FanOutProjection(in_channels=0, out_channels=32)

    def test_init_invalid_out_channels(self):
        with self.assertRaises(AssertionError):
            FanOutProjection(in_channels=3, out_channels=2)

    def test_forward_output_shape(self):
        proj = FanOutProjection(in_channels=3, out_channels=32)
        x = torch.randn(1, 3, 32, 32)
        z = proj.forward(x)
        self.assertEqual(z.shape, (1, 32, 32, 32))

    def test_add_weight_norms(self):
        proj = FanOutProjection(in_channels=3, out_channels=32)
        proj.add_weight_norms()
        x = torch.randn(1, 3, 32, 32)
        z = proj.forward(x)
        self.assertEqual(z.shape, (1, 32, 32, 32))

    def test_add_lora_adapters(self):
        proj = FanOutProjection(in_channels=3, out_channels=32)
        proj.add_lora_adapters(rank=4, alpha=1.0)
        x = torch.randn(1, 3, 32, 32)
        z = proj.forward(x)
        self.assertEqual(z.shape, (1, 32, 32, 32))


class TestUNet(unittest.TestCase):
    def test_init_valid_layers(self):
        unet = UNet(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        self.assertIsNotNone(unet.encoder)
        self.assertIsNotNone(unet.decoder)

    def test_init_invalid_primary_layers(self):
        with self.assertRaises(AssertionError):
            UNet(
                primary_channels=32,
                primary_layers=1,
                secondary_channels=64,
                secondary_layers=2,
                tertiary_channels=128,
                tertiary_layers=2,
                quaternary_channels=256,
                quaternary_layers=2,
                hidden_ratio=4,
            )

    def test_forward_output_shape(self):
        unet = UNet(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        x = torch.randn(1, 32, 32, 32)
        z, z_qa = unet.forward(x)
        self.assertEqual(z.shape, x.shape)
        self.assertEqual(z_qa.shape[0], 1)

    def test_enable_activation_checkpointing(self):
        unet = UNet(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        unet.enable_activation_checkpointing()
        x = torch.randn(1, 32, 32, 32)
        z, _ = unet.forward(x)
        self.assertEqual(z.shape, x.shape)


class TestEncoder(unittest.TestCase):
    def test_init_valid_layers(self):
        encoder = Encoder(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        self.assertIsNotNone(encoder.stage1)

    def test_init_invalid_primary_layers(self):
        with self.assertRaises(AssertionError):
            Encoder(
                primary_channels=32,
                primary_layers=0,
                secondary_channels=64,
                secondary_layers=2,
                tertiary_channels=128,
                tertiary_layers=2,
                quaternary_channels=256,
                quaternary_layers=2,
                hidden_ratio=4,
            )

    def test_forward_output_shapes(self):
        encoder = Encoder(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        x = torch.randn(1, 32, 32, 32)
        z1, z2, z3, z4, z_qa = encoder.forward(x)
        self.assertEqual(z1.shape, (1, 32, 32, 32))
        self.assertEqual(z2.shape, (1, 64, 16, 16))
        self.assertEqual(z3.shape, (1, 128, 8, 8))
        self.assertEqual(z4.shape, (1, 256, 4, 4))
        self.assertEqual(z_qa.shape[0], 1)

    def test_add_qa_head(self):
        encoder = Encoder(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        encoder.add_qa_head(num_features=5)
        x = torch.randn(1, 32, 32, 32)
        z1, z2, z3, z4, z_qa = encoder.forward(x)
        self.assertEqual(z_qa.shape, (1, 5))

    def test_remove_qa_head(self):
        encoder = Encoder(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        encoder.add_qa_head(num_features=5)
        encoder.remove_qa_head()
        x = torch.randn(1, 32, 32, 32)
        z1, z2, z3, z4, z_qa = encoder.forward(x)
        self.assertEqual(z_qa.shape[0], 1)

    def test_add_weight_norms(self):
        encoder = Encoder(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        encoder.add_weight_norms()
        x = torch.randn(1, 32, 32, 32)
        z1, z2, z3, z4, _ = encoder.forward(x)
        self.assertEqual(z1.shape, (1, 32, 32, 32))


class TestEncoderBlock(unittest.TestCase):
    def test_forward_output_shape(self):
        block = EncoderBlock(num_channels=32, hidden_ratio=4)
        x = torch.randn(1, 32, 16, 16)
        z = block.forward(x)
        self.assertEqual(z.shape, (1, 32, 16, 16))

    def test_add_weight_norms(self):
        block = EncoderBlock(num_channels=32, hidden_ratio=4)
        block.add_weight_norms()
        x = torch.randn(1, 32, 16, 16)
        z = block.forward(x)
        self.assertEqual(z.shape, (1, 32, 16, 16))


class TestDecoder(unittest.TestCase):
    def test_init_valid_layers(self):
        decoder = Decoder(
            primary_channels=256,
            primary_layers=2,
            secondary_channels=128,
            secondary_layers=2,
            tertiary_channels=64,
            tertiary_layers=2,
            quaternary_channels=32,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        self.assertIsNotNone(decoder.stage1)

    def test_init_invalid_primary_layers(self):
        with self.assertRaises(AssertionError):
            Decoder(
                primary_channels=256,
                primary_layers=0,
                secondary_channels=128,
                secondary_layers=2,
                tertiary_channels=64,
                tertiary_layers=2,
                quaternary_channels=32,
                quaternary_layers=2,
                hidden_ratio=4,
            )

    def test_forward_output_shape(self):
        decoder = Decoder(
            primary_channels=256,
            primary_layers=2,
            secondary_channels=128,
            secondary_layers=2,
            tertiary_channels=64,
            tertiary_layers=2,
            quaternary_channels=32,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        x1 = torch.randn(1, 256, 4, 4)
        x2 = torch.randn(1, 128, 8, 8)
        x3 = torch.randn(1, 64, 16, 16)
        x4 = torch.randn(1, 32, 32, 32)
        z = decoder.forward(x1, x2, x3, x4)
        self.assertEqual(z.shape, (1, 32, 32, 32))

    def test_crop_feature_maps_larger(self):
        decoder = Decoder(
            primary_channels=256,
            primary_layers=2,
            secondary_channels=128,
            secondary_layers=2,
            tertiary_channels=64,
            tertiary_layers=2,
            quaternary_channels=32,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        x = torch.randn(1, 32, 16, 16)
        size = (8, 8)
        z = decoder.crop_feature_maps(x, size)
        self.assertEqual(z.shape, (1, 32, 8, 8))

    def test_crop_feature_maps_smaller(self):
        decoder = Decoder(
            primary_channels=256,
            primary_layers=2,
            secondary_channels=128,
            secondary_layers=2,
            tertiary_channels=64,
            tertiary_layers=2,
            quaternary_channels=32,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        x = torch.randn(1, 32, 4, 4)
        size = (8, 8)
        z = decoder.crop_feature_maps(x, size)
        self.assertEqual(z.shape, (1, 32, 8, 8))


class TestDecoderBlock(unittest.TestCase):
    def test_forward_output_shape(self):
        block = DecoderBlock(num_channels=32, hidden_ratio=4)
        x = torch.randn(1, 32, 16, 16)
        z = block.forward(x)
        self.assertEqual(z.shape, (1, 32, 16, 16))


class TestInvertedBottleneck(unittest.TestCase):
    def test_init_valid(self):
        bottleneck = InvertedBottleneck(num_channels=32, hidden_ratio=4)
        self.assertIsNotNone(bottleneck.conv1)
        self.assertIsNotNone(bottleneck.conv2)

    def test_init_invalid_channels(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(num_channels=0, hidden_ratio=4)

    def test_init_invalid_hidden_ratio(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(num_channels=32, hidden_ratio=3)

    def test_forward_output_shape(self):
        bottleneck = InvertedBottleneck(num_channels=32, hidden_ratio=4)
        x = torch.randn(1, 32, 16, 16)
        z = bottleneck.forward(x)
        self.assertEqual(z.shape, (1, 32, 16, 16))

    def test_add_weight_norms(self):
        bottleneck = InvertedBottleneck(num_channels=32, hidden_ratio=4)
        bottleneck.add_weight_norms()
        x = torch.randn(1, 32, 16, 16)
        z = bottleneck.forward(x)
        self.assertEqual(z.shape, (1, 32, 16, 16))


class TestResidualConnection(unittest.TestCase):
    def test_forward_same_shape(self):
        skip = ResidualConnection()
        x = torch.randn(1, 32, 16, 16)
        z = torch.randn(1, 32, 16, 16)
        out = skip.forward(x, z)
        self.assertEqual(out.shape, (1, 32, 16, 16))
        torch.testing.assert_close(out, x + z)

    def test_forward_different_shape_raises(self):
        skip = ResidualConnection()
        x = torch.randn(1, 32, 16, 16)
        z = torch.randn(1, 32, 8, 8)
        with self.assertRaises(AssertionError):
            skip.forward(x, z)


class TestPixelCrush(unittest.TestCase):
    def test_init_valid(self):
        crush = PixelCrush(in_channels=32, out_channels=64, crush_factor=2)
        self.assertIsNotNone(crush.conv)

    def test_init_invalid_in_channels(self):
        with self.assertRaises(AssertionError):
            PixelCrush(in_channels=0, out_channels=64, crush_factor=2)

    def test_init_invalid_out_channels(self):
        with self.assertRaises(AssertionError):
            PixelCrush(in_channels=32, out_channels=0, crush_factor=2)

    def test_init_invalid_crush_factor(self):
        with self.assertRaises(AssertionError):
            PixelCrush(in_channels=32, out_channels=64, crush_factor=5)

    def test_forward_output_shape(self):
        crush = PixelCrush(in_channels=32, out_channels=64, crush_factor=2)
        x = torch.randn(1, 32, 16, 16)
        z = crush.forward(x)
        self.assertEqual(z.shape, (1, 64, 8, 8))

    def test_add_weight_norms(self):
        crush = PixelCrush(in_channels=32, out_channels=64, crush_factor=2)
        crush.add_weight_norms()
        x = torch.randn(1, 32, 16, 16)
        z = crush.forward(x)
        self.assertEqual(z.shape, (1, 64, 8, 8))


class TestSubpixelConv2d(unittest.TestCase):
    def test_init_valid(self):
        subpixel = SubpixelConv2d(in_channels=32, out_channels=64, upscale_ratio=2)
        self.assertIsNotNone(subpixel.conv)

    def test_init_invalid_in_channels(self):
        with self.assertRaises(AssertionError):
            SubpixelConv2d(in_channels=0, out_channels=64, upscale_ratio=2)

    def test_init_invalid_out_channels(self):
        with self.assertRaises(AssertionError):
            SubpixelConv2d(in_channels=32, out_channels=0, upscale_ratio=2)

    def test_init_invalid_upscale_ratio(self):
        with self.assertRaises(AssertionError):
            SubpixelConv2d(in_channels=32, out_channels=64, upscale_ratio=5)

    def test_forward_output_shape(self):
        subpixel = SubpixelConv2d(in_channels=32, out_channels=3, upscale_ratio=2)
        x = torch.randn(1, 32, 8, 8)
        z = subpixel.forward(x)
        self.assertEqual(z.shape, (1, 3, 16, 16))

    def test_forward_output_shape_x4(self):
        subpixel = SubpixelConv2d(in_channels=32, out_channels=3, upscale_ratio=4)
        x = torch.randn(1, 32, 4, 4)
        z = subpixel.forward(x)
        self.assertEqual(z.shape, (1, 3, 16, 16))

    def test_add_weight_norms(self):
        subpixel = SubpixelConv2d(in_channels=32, out_channels=3, upscale_ratio=2)
        subpixel.add_weight_norms()
        x = torch.randn(1, 32, 8, 8)
        z = subpixel.forward(x)
        self.assertEqual(z.shape, (1, 3, 16, 16))


class TestQualityAssessor(unittest.TestCase):
    def test_init_valid(self):
        assessor = QualityAssessor(num_channels=256, num_labels=5)
        self.assertIsNotNone(assessor.conv)

    def test_init_invalid_num_labels(self):
        with self.assertRaises(AssertionError):
            QualityAssessor(num_channels=256, num_labels=0)

    def test_forward_output_shape(self):
        assessor = QualityAssessor(num_channels=256, num_labels=5)
        x = torch.randn(1, 256, 4, 4)
        z = assessor.forward(x)
        self.assertEqual(z.shape, (1, 5))

    def test_add_weight_norms(self):
        assessor = QualityAssessor(num_channels=256, num_labels=5)
        assessor.add_weight_norms()
        x = torch.randn(1, 256, 4, 4)
        z = assessor.forward(x)
        self.assertEqual(z.shape, (1, 5))


class TestSuperResolver(unittest.TestCase):
    def test_init_valid(self):
        resolver = SuperResolver(in_channels=32, upscale_ratio=2)
        self.assertIsNotNone(resolver.upscale)

    def test_init_invalid_upscale_ratio(self):
        with self.assertRaises(AssertionError):
            SuperResolver(in_channels=32, upscale_ratio=5)

    def test_forward_output_shape(self):
        resolver = SuperResolver(in_channels=32, upscale_ratio=2)
        x = torch.randn(1, 32, 16, 16)
        z = resolver.forward(x)
        self.assertEqual(z.shape, (1, 3, 32, 32))


class TestChannelLoRA(unittest.TestCase):
    def test_init_valid(self):
        conv = Conv2d(32, 64, kernel_size=3, padding=1)
        lora = ChannelLoRA(layer=conv, rank=4, alpha=1.0)
        self.assertIsNotNone(lora.lora_a)
        self.assertIsNotNone(lora.lora_b)

    def test_init_invalid_rank(self):
        conv = Conv2d(32, 64, kernel_size=3, padding=1)
        with self.assertRaises(AssertionError):
            ChannelLoRA(layer=conv, rank=0, alpha=1.0)

    def test_init_invalid_alpha(self):
        conv = Conv2d(32, 64, kernel_size=3, padding=1)
        with self.assertRaises(AssertionError):
            ChannelLoRA(layer=conv, rank=4, alpha=0.0)

    def test_forward_output_shape(self):
        conv = Conv2d(32, 64, kernel_size=3, padding=1)
        lora = ChannelLoRA(layer=conv, rank=4, alpha=1.0)
        w = torch.randn(64, 32, 3, 3)
        z = lora.forward(w)
        self.assertEqual(z.shape, w.shape)


class TestBouncer(unittest.TestCase):
    def test_from_preconfigured_small(self):
        bouncer = Bouncer.from_preconfigured("small")
        self.assertIsNotNone(bouncer.stem)
        self.assertIsNotNone(bouncer.detector)

    def test_from_preconfigured_medium(self):
        bouncer = Bouncer.from_preconfigured("medium")
        self.assertIsNotNone(bouncer.stem)

    def test_from_preconfigured_large(self):
        bouncer = Bouncer.from_preconfigured("large")
        self.assertIsNotNone(bouncer.stem)

    def test_from_preconfigured_invalid(self):
        with self.assertRaises(ValueError):
            Bouncer.from_preconfigured("invalid")

    def test_forward_output_shapes(self):
        bouncer = Bouncer.from_preconfigured("small")
        x = torch.randn(1, 3, 64, 64)
        z1, z2, z3, z4, z5 = bouncer.forward(x)
        self.assertEqual(z1.shape[0], 1)
        self.assertEqual(z2.shape[0], 1)
        self.assertEqual(z3.shape[0], 1)
        self.assertEqual(z4.shape[0], 1)
        self.assertEqual(z5.shape[0], 1)

    def test_predict(self):
        bouncer = Bouncer.from_preconfigured("small")
        bouncer.eval()
        x = torch.randn(1, 3, 64, 64)
        z = bouncer.predict(x)
        self.assertEqual(z.shape[0], 1)

    def test_add_spectral_norms(self):
        bouncer = Bouncer.from_preconfigured("small")
        bouncer.add_spectral_norms()
        x = torch.randn(1, 3, 64, 64)
        z1, z2, z3, z4, z5 = bouncer.forward(x)
        self.assertEqual(z1.shape[0], 1)


class TestFeatureDetector(unittest.TestCase):
    def test_init_valid(self):
        detector = FeatureDetector(
            primary_channels=64,
            primary_layers=2,
            secondary_channels=128,
            secondary_layers=2,
            tertiary_channels=256,
            tertiary_layers=2,
            quaternary_channels=512,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        self.assertIsNotNone(detector.stage1)

    def test_init_invalid_primary_layers(self):
        with self.assertRaises(AssertionError):
            FeatureDetector(
                primary_channels=64,
                primary_layers=0,
                secondary_channels=128,
                secondary_layers=2,
                tertiary_channels=256,
                tertiary_layers=2,
                quaternary_channels=512,
                quaternary_layers=2,
                hidden_ratio=4,
            )

    def test_forward_output_shapes(self):
        detector = FeatureDetector(
            primary_channels=64,
            primary_layers=2,
            secondary_channels=128,
            secondary_layers=2,
            tertiary_channels=256,
            tertiary_layers=2,
            quaternary_channels=512,
            quaternary_layers=2,
            hidden_ratio=4,
        )
        x = torch.randn(1, 64, 32, 32)
        z1, z2, z3, z4 = detector.forward(x)
        self.assertEqual(z1.shape, (1, 64, 32, 32))
        self.assertEqual(z2.shape, (1, 128, 16, 16))
        self.assertEqual(z3.shape, (1, 256, 8, 8))
        self.assertEqual(z4.shape, (1, 512, 4, 4))


class TestDetectorBlock(unittest.TestCase):
    def test_init_valid(self):
        block = DetectorBlock(num_channels=64, hidden_ratio=4)
        self.assertIsNotNone(block.conv1)

    def test_init_invalid_channels(self):
        with self.assertRaises(AssertionError):
            DetectorBlock(num_channels=0, hidden_ratio=4)

    def test_init_invalid_hidden_ratio(self):
        with self.assertRaises(AssertionError):
            DetectorBlock(num_channels=64, hidden_ratio=3)

    def test_forward_output_shape(self):
        block = DetectorBlock(num_channels=64, hidden_ratio=4)
        x = torch.randn(1, 64, 16, 16)
        z = block.forward(x)
        self.assertEqual(z.shape, (1, 64, 16, 16))

    def test_add_spectral_norms(self):
        block = DetectorBlock(num_channels=64, hidden_ratio=4)
        block.add_spectral_norms()
        x = torch.randn(1, 64, 16, 16)
        z = block.forward(x)
        self.assertEqual(z.shape, (1, 64, 16, 16))


class TestDepthwiseSeparableConv2d(unittest.TestCase):
    def test_init_valid(self):
        conv = DepthwiseSeparableConv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.assertIsNotNone(conv.depthwise)
        self.assertIsNotNone(conv.pointwise)

    def test_init_invalid_in_channels(self):
        with self.assertRaises(AssertionError):
            DepthwiseSeparableConv2d(
                in_channels=0, out_channels=64, kernel_size=3, padding=1
            )

    def test_init_invalid_out_channels(self):
        with self.assertRaises(AssertionError):
            DepthwiseSeparableConv2d(
                in_channels=32, out_channels=0, kernel_size=3, padding=1
            )

    def test_init_invalid_kernel_size(self):
        with self.assertRaises(AssertionError):
            DepthwiseSeparableConv2d(
                in_channels=32, out_channels=64, kernel_size=0, padding=1
            )

    def test_init_invalid_padding(self):
        with self.assertRaises(AssertionError):
            DepthwiseSeparableConv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=-1
            )

    def test_forward_output_shape(self):
        conv = DepthwiseSeparableConv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        x = torch.randn(1, 32, 16, 16)
        z = conv.forward(x)
        self.assertEqual(z.shape, (1, 64, 16, 16))

    def test_add_spectral_norms(self):
        conv = DepthwiseSeparableConv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        conv.add_spectral_norms()
        x = torch.randn(1, 32, 16, 16)
        z = conv.forward(x)
        self.assertEqual(z.shape, (1, 64, 16, 16))


class TestPatchDiscriminator(unittest.TestCase):
    def test_forward_output_shape(self):
        discriminator = PatchDiscriminator(num_channels=512)
        x = torch.randn(1, 512, 4, 4)
        z = discriminator.forward(x)
        self.assertEqual(z.shape[0], 1)

    def test_add_spectral_norms(self):
        discriminator = PatchDiscriminator(num_channels=512)
        discriminator.add_spectral_norms()
        x = torch.randn(1, 512, 4, 4)
        z = discriminator.forward(x)
        self.assertEqual(z.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
