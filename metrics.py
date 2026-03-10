import torch

from torch import Tensor

from torch.nn import Module

from torchmetrics.classification import BinaryPrecision, BinaryRecall


class PatchF1Score(Module):
    """Computes the F1 score on patches."""

    def __init__(self):
        super().__init__()

        self.precision_metric = BinaryPrecision()
        self.recall_metric = BinaryRecall()

    def update(self, y_pred_fake: Tensor, y_pred_real: Tensor) -> None:
        """
        Args:
            y_pred_fake: (B, 1, H, W) - patch predictions for fake images
            y_pred_real: (B, 1, H, W) - patch predictions for real images
        """

        y_pred_fake = y_pred_fake.flatten()  # (B * H * W)
        y_pred_real = y_pred_real.flatten()  # (B * H * W)

        y_pred = torch.cat([y_pred_fake, y_pred_real], dim=0)

        y_fake = torch.zeros(y_pred_fake.size(0), device=y_pred_fake.device)
        y_real = torch.ones(y_pred_real.size(0), device=y_pred_real.device)

        y = torch.cat([y_fake, y_real], dim=0)

        self.precision_metric.update(y_pred, y)
        self.recall_metric.update(y_pred, y)

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()

        if precision + recall == 0:
            f1_score = torch.tensor(0.0, device=precision.device)
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score, precision, recall

    def reset(self) -> None:
        self.precision_metric.reset()
        self.recall_metric.reset()
