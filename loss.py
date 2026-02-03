import torch

from torch import Tensor

from torch.nn import Module, MSELoss, BCEWithLogitsLoss, Parameter

from torchvision.models import vgg19, VGG19_Weights

from src.ultrazoom.model import Bouncer


class VGGLoss(Module):
    """
    A perceptual loss based on the L2 distance between low and high-level VGG19
    embeddings of the predicted and target image.
    """

    def __init__(self):
        super().__init__()

        model = vgg19(weights=VGG19_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        self.vgg22 = model.features[0:9]
        self.vgg54 = model.features[9:36]

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
    Relativistic average BCE with logits loss for generative adversarial network training.
    """

    def __init__(self):
        super().__init__()

        self.bce = BCEWithLogitsLoss()

    def forward(
        self,
        y_pred_real: Tensor,
        y_pred_fake: Tensor,
        y_real: Tensor,
        y_fake: Tensor,
    ) -> Tensor:
        y_pred_real_hat = y_pred_real - y_pred_fake.mean()
        y_pred_fake_hat = y_pred_fake - y_pred_real.mean()

        y_pred = torch.cat((y_pred_real_hat, y_pred_fake_hat))
        y = torch.cat((y_real, y_fake))

        loss = self.bce.forward(y_pred, y)

        return loss


class WassersteinLoss(Module):
    """
    Wasserstein loss with gradient penalty for generative adversarial network training.
    """

    def __init__(self, critic: Bouncer, penalty_lambda: float = 10.0):
        super().__init__()

        assert penalty_lambda > 0.0, "Penalty lambda must be positive."

        self.critic = critic
        self.penalty_lambda = penalty_lambda

    def compute_gradient_penalty(self, y_orig: Tensor, u_pred_sr: Tensor) -> Tensor:
        """
        Compute gradient penalty for Lipschitz constraint enforcement.

        Args:
            y_orig: Original high-resolution images.
            u_pred_sr: Super-resolved images from the upscaler.

        Returns:
            Gradient penalty tensor.
        """

        # Random interpolation
        alpha = torch.rand(y_orig.size(0), 1, 1, 1, device=y_orig.device)

        interpolated = alpha * y_orig + (1 - alpha) * u_pred_sr.detach()

        interpolated.requires_grad_(True)

        _, _, _, _, d_interpolated = self.critic.forward(interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )

        penalty = ((gradients[0].norm(dim=1) - 1) ** 2).mean()

        return self.penalty_lambda * penalty

    def critic_loss(
        self,
        y_pred_real: Tensor,
        y_pred_fake: Tensor,
        y_orig: Tensor,
        u_pred_sr: Tensor,
    ) -> Tensor:
        """
        Compute critic loss: maximize E[D(real)] - E[D(fake)] + gradient penalty.

        Args:
            y_pred_real: Critic output for real images.
            y_pred_fake: Critic output for fake images.
            y_orig: Original high-resolution images.
            u_pred_sr: Super-resolved images from the upscaler.

        Returns:
            Combined Wasserstein loss with gradient penalty.
        """

        assert (
            y_pred_real.shape == y_pred_fake.shape
        ), "Real and fake predictions must have the same shape"

        assert (
            y_orig.shape == u_pred_sr.shape
        ), "Original and super-resolved images must have same shape"

        loss = torch.mean(y_pred_fake) - torch.mean(y_pred_real)

        gradient_penalty = self.compute_gradient_penalty(y_orig, u_pred_sr)

        return loss + gradient_penalty

    def generator_loss(self, y_pred_fake: Tensor) -> Tensor:
        """
        Compute generator loss: minimize -E[D(fake)].

        Args:
            y_pred_fake: Critic output for fake images.

        Returns:
            Generator adversarial loss.
        """

        return -torch.mean(y_pred_fake)


class DegredationAwareFocalLossWeighting(Module):
    """
    A loss weighting that focuses the loss on samples that are harder to upscale due to having
    higher amounts of degradation present in the input.
    """

    def __init__(self, gamma: float):
        super().__init__()

        self.gamma = gamma

    def forward(self, y_qa: Tensor) -> Tensor:
        qa_norms = y_qa.norm(dim=1, keepdim=True)

        weights = (1 + qa_norms) ** self.gamma

        return weights


class BalancedMultitaskLoss(Module):
    """A dynamic multitask loss weighting where each task contributes equally."""

    def __init__(self):
        super().__init__()

    def forward(self, losses: Tensor) -> Tensor:
        combined_loss = losses / losses.detach()

        combined_loss = combined_loss.sum()

        return combined_loss


class AdaptiveMultitaskLoss(Module):
    """
    Adaptive loss weighting using homoscedastic i.e. task-dependent uncertainty as a training signal.
    """

    def __init__(self, num_losses: int):
        super().__init__()

        assert num_losses > 0, "Number of losses must be positive"

        self.log_sigmas = Parameter(torch.zeros(num_losses))

        self.num_losses = num_losses

    @property
    def loss_weights(self) -> Tensor:
        """
        Get current loss weights based on learned uncertainties.

        Returns:
            Tensor of loss weights for each task.
        """

        weights = torch.exp(-2.0 * self.log_sigmas)

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

        # Regularization term to prevent task weight collapse.
        weighted_losses += self.log_sigmas

        combined_loss = weighted_losses.sum()

        return combined_loss
