import torch

from torch import Tensor

from torch.nn import Module

from torchmetrics.classification import BinaryPrecision, BinaryRecall


class RelativisticF1Score(Module):
    """Computes the F1 score using relativistic mean predictions."""

    def __init__(self):
        super().__init__()

        self.precision_metric = BinaryPrecision()

        self.recall_metric = BinaryRecall()

    def update(self, y_pred_fake: Tensor, y_pred_real: Tensor) -> None:
        y_fake = torch.full((y_pred_fake.size(0), 1), 0.0)
        y_real = torch.full((y_pred_real.size(0), 1), 1.0)

        y_fake = y_fake.to(y_pred_fake.device)
        y_real = y_real.to(y_pred_real.device)

        y_pred_fake_hat = y_pred_fake - y_pred_real.mean()
        y_pred_real_hat = y_pred_real - y_pred_fake.mean()

        y_pred = torch.cat((y_pred_fake_hat, y_pred_real_hat), dim=0)
        labels = torch.cat((y_fake, y_real), dim=0)

        self.precision_metric.update(y_pred, labels)
        self.recall_metric.update(y_pred, labels)

    def compute(self) -> tuple[Tensor, ...]:
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


class EarthMoverMetric(Module):
    """Computes the Wasserstein distance (Earth Mover's Distance) between real and fake critic scores."""

    def __init__(self):
        super().__init__()

        self.reset()

    def update(self, y_pred_fake: Tensor, y_pred_real: Tensor) -> None:
        self.sum_real += y_pred_real.sum()
        self.sum_fake += y_pred_fake.sum()

        self.count += y_pred_real.size(0)

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(0.0)

        mean_real = self.sum_real / self.count
        mean_fake = self.sum_fake / self.count

        distance = mean_real - mean_fake

        return distance

    def reset(self) -> None:
        self.sum_real = torch.tensor(0.0)
        self.sum_fake = torch.tensor(0.0)

        self.count = 0
