import unittest
import torch
import torch.nn as nn
from torch import Tensor

from loss import RelativisticBCELoss


class TestRelativisticBCELoss(unittest.TestCase):
    """Test suite for RelativisticBCELoss class"""

    def setUp(self):
        """Initialize test fixtures before each test"""
        self.loss_module = RelativisticBCELoss()

    def test_initialization(self):
        """Test that loss module initializes correctly"""
        self.assertIsInstance(self.loss_module, nn.Module)

        self.assertIsInstance(self.loss_module.bce, nn.BCEWithLogitsLoss)

    def test_forward_pass_basic(self):
        """Test forward pass with basic valid inputs"""
        batch_size = 4
        num_features = 64

        y_pred_real = torch.randn(batch_size, num_features)
        y_pred_fake = torch.randn(batch_size, num_features)
        y_real = torch.rand(batch_size, num_features)
        y_fake = torch.rand(batch_size, num_features)

        output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output, Tensor)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.dim(), 0)

    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample"""
        num_features = 32

        y_pred_real = torch.randn(1, num_features)
        y_pred_fake = torch.randn(1, num_features)
        y_real = torch.rand(1, num_features)
        y_fake = torch.rand(1, num_features)

        output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.dim(), 0)

    def test_backward_pass_gradient_computation(self):
        """Test backward pass and gradient computation"""
        batch_size = 3
        num_features = 16

        y_pred_real = torch.randn(batch_size, num_features, requires_grad=True)
        y_pred_fake = torch.randn(batch_size, num_features, requires_grad=True)
        y_real = torch.rand(batch_size, num_features)
        y_fake = torch.rand(batch_size, num_features)

        output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        output.backward()

        self.assertIsNotNone(y_pred_real.grad)
        self.assertIsNotNone(y_pred_fake.grad)
        self.assertTrue(torch.is_tensor(y_pred_real.grad))
        self.assertTrue(torch.is_tensor(y_pred_fake.grad))
        self.assertTrue(y_pred_real.grad.requires_grad == False)

    def test_relativistic_mean_calculation(self):
        """Test relativistic mean calculation logic"""
        batch_size = 4

        # Case 1: Different means for real and fake
        y_pred_real = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], requires_grad=True
        )
        y_pred_fake = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], requires_grad=True
        )
        y_real = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True
        )
        y_fake = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True
        )

        output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output, Tensor)
        self.assertTrue(torch.isfinite(output))

    def test_different_tensor_shapes(self):
        """Test forward pass with different tensor shapes"""
        shapes = [(1, 10), (2, 10), (4, 10), (4, 20), (4, 32), (16, 64), (32, 128)]

        for shape in shapes:
            y_pred_real = torch.randn(*shape)
            y_pred_fake = torch.randn(*shape)
            y_real = torch.rand(*shape)
            y_fake = torch.rand(*shape)

            output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

            self.assertIsInstance(output, Tensor)
            self.assertEqual(output.dim(), 0)

    def test_various_batch_sizes(self):
        """Test forward pass with various batch sizes"""
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            y_pred_real = torch.randn(batch_size, 16)
            y_pred_fake = torch.randn(batch_size, 16)
            y_real = torch.rand(batch_size, 16)
            y_fake = torch.rand(batch_size, 16)

            output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

            self.assertIsInstance(output, Tensor)
            self.assertEqual(output.dim(), 0)
            self.assertTrue(torch.isfinite(output))

    def test_device_placement(self):
        """Test computation on different devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test on CPU
        y_pred_real = torch.randn(4, 16)
        y_pred_fake = torch.randn(4, 16)
        y_real = torch.rand(4, 16)
        y_fake = torch.rand(4, 16)

        output_cpu = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output_cpu, Tensor)
        self.assertEqual(output_cpu.device.type, "cpu")

        # Test on GPU
        y_pred_real = torch.randn(4, 16).cuda()
        y_pred_fake = torch.randn(4, 16).cuda()
        y_real = torch.rand(4, 16).cuda()
        y_fake = torch.rand(4, 16).cuda()

        output_gpu = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output_gpu, Tensor)
        self.assertEqual(output_gpu.device.type, "cuda")

    def test_different_dtypes(self):
        """Test with different tensor data types"""
        dtypes = [torch.float32, torch.float64]

        for dtype in dtypes:
            y_pred_real = torch.randn(4, 16, dtype=dtype)
            y_pred_fake = torch.randn(4, 16, dtype=dtype)
            y_real = torch.rand(4, 16, dtype=dtype)
            y_fake = torch.rand(4, 16, dtype=dtype)

            output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

            self.assertIsInstance(output, Tensor)
            self.assertEqual(output.dim(), 0)

    def test_input_validation_shape_mismatch(self):
        """Test error handling for shape mismatches. RelativisticBCELoss raises ValueError from binary_cross_entropy_with_logits."""
        # Different batch sizes and feature dimensions cause various exceptions
        with self.assertRaises((RuntimeError, ValueError)):
            y_pred_real = torch.randn(4, 16)
            y_pred_fake = torch.randn(2, 8)
            y_real = torch.rand(4, 16)
            y_fake = torch.rand(4, 16)
            self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        # Feature dimension mismatch causes ValueError from binary_cross_entropy_with_logits
        with self.assertRaises((RuntimeError, ValueError)):
            y_pred_real = torch.randn(4, 16)
            y_pred_fake = torch.randn(4, 8)
            y_real = torch.rand(4, 16)
            y_fake = torch.rand(4, 8)
            self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

    def test_edge_cases_zero_values(self):
        """Test with zero values in input tensors"""
        y_pred_real = torch.zeros(4, 16)
        y_pred_fake = torch.zeros(4, 16)
        y_real = torch.zeros(4, 16)
        y_fake = torch.zeros(4, 16)

        output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output, Tensor)
        self.assertTrue(torch.isfinite(output))

    def test_edge_cases_constant_values(self):
        """Test with constant values"""
        y_pred_real = torch.ones(4, 16) * 2.0
        y_pred_fake = torch.ones(4, 16) * 1.0
        y_real = torch.ones(4, 16) * 1.0
        y_fake = torch.ones(4, 16) * 0.0

        output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output, Tensor)
        self.assertTrue(torch.isfinite(output))

    def test_large_extreme_values(self):
        """Test with large positive/negative values"""
        y_pred_real = torch.randn(4, 16) * 1e6
        y_pred_fake = torch.randn(4, 16) * 1e6
        y_real = torch.rand(4, 16)
        y_fake = torch.rand(4, 16)

        output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output, Tensor)
        self.assertTrue(torch.isfinite(output))

    def test_tensor_type_enforcement(self):
        """Test that inputs are properly converted to tensor types"""
        y_pred_real = torch.randn(4, 16)
        y_pred_fake = torch.randn(4, 16)
        y_real = torch.rand(4, 16)
        y_fake = torch.rand(4, 16)

        # These should work with basic types
        output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output, Tensor)

    def test_integration_with_bce_with_logits_loss(self):
        """Test integration with BCEWithLogitsLoss wrapper"""
        batch_size = 4
        num_features = 16

        y_pred_real = torch.randn(batch_size, num_features)
        y_pred_fake = torch.randn(batch_size, num_features)
        y_real = torch.rand(batch_size, num_features)
        y_fake = torch.rand(batch_size, num_features)

        outputs = [
            self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake),
            self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake),
            self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake),
        ]

        self.assertEqual(len(outputs), 3)
        for output in outputs:
            self.assertIsInstance(output, Tensor)
            self.assertTrue(torch.isfinite(output))

    def test_gradient_duality(self):
        """Test gradient computation and backpropagation"""
        for i in range(3):
            batch_size = 4
            num_features = 8

            y_pred_real = torch.randn(batch_size, num_features, requires_grad=True)
            y_pred_fake = torch.randn(batch_size, num_features, requires_grad=True)
            y_real = torch.rand(batch_size, num_features)
            y_fake = torch.rand(batch_size, num_features)

            output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

            output.backward()

            # Check gradients are computed
            self.assertIsNotNone(y_pred_real.grad)
            self.assertIsNotNone(y_pred_fake.grad)

            # Check gradients have correct shape
            self.assertEqual(y_pred_real.grad.shape, y_pred_real.shape)
            self.assertEqual(y_pred_fake.grad.shape, y_pred_fake.shape)

    def test_batch_statistics_computation(self):
        """Test batch mean computation for relativistic differences"""

        y_pred_real = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0],
                [8.0, 9.0, 1.0],
            ],
            requires_grad=True,
        )

        y_pred_fake = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
                [0.8, 0.9, 0.1],
            ],
            requires_grad=True,
        )

        y_real = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            requires_grad=True,
        )

        y_fake = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            requires_grad=True,
        )

        output = self.loss_module(y_pred_real, y_pred_fake, y_real, y_fake)

        self.assertIsInstance(output, Tensor)
        self.assertTrue(torch.isfinite(output))


if __name__ == "__main__":
    unittest.main()
