import torch

from torch import Tensor

from torch.nn import Module, MSELoss, BCEWithLogitsLoss, Parameter, Sequential, Buffer

from torchvision.models import vgg19, VGG19_Weights, VGG


class VGGLoss(Module):
    """
    A perceptual loss based on the L2 distance between low and high-level VGG19
    embeddings of the predicted and target image.
    """

    def __init__(self):
        super().__init__()

        model: VGG = vgg19(weights=VGG19_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        # Compensate for incorrect Torchvision typehinting.
        features: Sequential = model.features

        self.vgg22 = features[0:9]
        self.vgg54 = features[9:36]

        self.mse = MSELoss()

    @property
    def num_params(self) -> int:
        num_params = 0

        for module in (self.vgg22, self.vgg54):
            num_params += sum(param.numel() for param in module.parameters())

        return num_params

    def forward(self, y_pred: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        y_pred_vgg22 = self.vgg22.forward(y_pred)
        y_vgg22 = self.vgg22.forward(y)

        vgg22_loss = self.mse.forward(y_pred_vgg22, y_vgg22)

        y_pred_vgg54 = self.vgg54.forward(y_pred_vgg22)
        y_vgg54 = self.vgg54.forward(y_vgg22)

        vgg54_loss = self.mse.forward(y_pred_vgg54, y_vgg54)

        return vgg22_loss, vgg54_loss


class RelativisticBCELoss(Module):
    """
    Relativistic average BCE with logits loss on patches for generative adversarial network training.
    """

    def __init__(self, real_label_jitter: float, fake_label_jitter: float):
        super().__init__()

        assert 0.0 <= real_label_jitter < 1.0, "Real label jitter must be in [0, 1)."
        assert 0.0 <= fake_label_jitter < 1.0, "Fake label jitter must be in [0, 1)."

        self.bce = BCEWithLogitsLoss()

        self.real_label_jitter = real_label_jitter
        self.fake_label_jitter = fake_label_jitter

    def forward_critic(self, y_pred_fake: Tensor, y_pred_real: Tensor) -> Tensor:
        """
        Compute critic loss.

        Args:
            y_pred_real: Critic output for real images (B, C, H, W).
            y_pred_fake: Critic output for fake images (B, C, H, W).
        """

        y_pred_fake = y_pred_fake.squeeze(1)  # (B, H, W)
        y_pred_real = y_pred_real.squeeze(1)  # (B, H, W)

        y_pred_fake_sigma = y_pred_fake.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        y_pred_real_sigma = y_pred_real.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)

        y_pred_fake = y_pred_fake - y_pred_real_sigma
        y_pred_real = y_pred_real - y_pred_fake_sigma

        y_pred = torch.cat((y_pred_fake, y_pred_real))

        if self.fake_label_jitter > 0.0:
            y_fake = torch.rand_like(y_pred_fake) * self.fake_label_jitter
        else:
            y_fake = torch.zeros_like(y_pred_fake)

        if self.real_label_jitter > 0.0:
            y_real = 1.0 - torch.rand_like(y_pred_real) * self.real_label_jitter
        else:
            y_real = torch.ones_like(y_pred_real)

        y = torch.cat((y_fake, y_real))

        loss = self.bce.forward(y_pred, y)

        return loss

    def forward_upscaler(self, y_pred_fake: Tensor, y_pred_real: Tensor) -> Tensor:
        """
        Compute generator loss.

        Args:
            y_pred_fake: Critic output for fake images (B, C, H, W).
            y_pred_real: Critic output for real images (B, C, H, W).
        """

        y_pred_fake = y_pred_fake.squeeze(1)  # (B, H, W)
        y_pred_real = y_pred_real.squeeze(1)  # (B, H, W)

        y_pred_real_sigma = y_pred_real.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)

        y_pred = y_pred_fake - y_pred_real_sigma

        y = torch.ones_like(y_pred)

        loss = self.bce.forward(y_pred, y)

        return loss


class R1GradientPenalty(Module):
    """
    R1 regularization penalty for generative adversarial network training that penalizes the
    gradient of the critic's output with respect to real images.
    """

    def __init__(self, gamma: float):
        super().__init__()

        assert gamma >= 0.0, "Gamma must be non-negative."

        self.gamma = torch.tensor(gamma)

    def forward(self, y_pred_real: Tensor, y_real: Tensor) -> Tensor:
        """
        Compute R1 regularization penalty.

        Args:
            y_pred_real: Critic output for real images.
            y_real: Real images corresponding to y_pred_real.
        """

        grad_outputs = torch.ones_like(y_pred_real)

        gradients = torch.autograd.grad(
            outputs=y_pred_real,
            inputs=y_real,
            grad_outputs=grad_outputs,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)

        norms = gradients.norm(2, dim=1).square().mean()

        penalty = 0.5 * self.gamma * norms

        return penalty


class AdaptiveMultitaskLoss(Module):
    """
    Adaptive loss weighting using homoscedastic i.e. task-dependent uncertainty as a training signal.
    """

    def __init__(self, num_losses: int, min_weight: float):
        super().__init__()

        assert num_losses > 0, "Number of losses must be positive"

        self.log_sigmas = Parameter(torch.zeros(num_losses))

        self.num_losses = num_losses
        self.min_weight = min_weight

    @property
    def loss_weights(self) -> Tensor:
        """
        Get current loss weights based on learned uncertainties.

        Returns:
            Tensor of loss weights for each task.
        """

        weights = torch.exp(-2.0 * self.log_sigmas)

        weights = weights.clamp(min=self.min_weight)

        return weights

    def forward(self, losses: Tensor) -> Tensor:
        """
        Compute task uncertainty-weighted combined loss.

        Args:
            losses: Tensor of individual loss values for each task.

        Returns:
            Combined task uncertainty-weighted loss.
        """

        assert (
            losses.size(0) == self.num_losses
        ), "Number of losses must match number of tasks."

        weighted_losses = 0.5 * self.loss_weights * losses

        regularized_losses = weighted_losses + self.log_sigmas

        combined_loss = regularized_losses.sum()

        return combined_loss


class WeightedMultitaskLoss(Module):
    """A multitask loss weighting where each task contributes based on a static scalar."""

    def __init__(self, weights: list[float]):
        super().__init__()

        num_losses = len(weights)

        assert num_losses > 0, "Number of losses must be positive."

        self.weights = Buffer(torch.tensor(weights, dtype=torch.float32))

        self.num_losses = num_losses

    def forward(self, losses: Tensor) -> Tensor:
        assert (
            losses.size(0) == self.num_losses
        ), "Number of losses must match number of tasks."

        weighted_losses = self.weights * losses

        combined_loss = weighted_losses.sum()

        return combined_loss
