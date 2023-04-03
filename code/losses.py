import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from pytorch_metric_learning.losses.supcon_loss import SupConLoss
from pytorch_metric_learning.reducers.do_nothing_reducer import DoNothingReducer


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduction = "none"

    def forward(self, logits, targets, mask):
        loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        loss = (loss * mask).mean()
        return loss


class CrossEntropy(nn.Module):
    def __init__(self, label_smoothing):
        super().__init__()
        self.reduction = "none"
        self.ce_loss = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.masked_ce_loss = MaskedCrossEntropyLoss()

    def forward(self, logits, targets, mask=None):
        if mask == None:
            loss = self.ce_loss(logits, targets)
        else:
            loss = self.masked_ce_loss(logits, targets, mask)
        return loss


class MaskedSupConLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.low = 0
        self.high = None
        self.reducer = DoNothingReducer()
        self.criterion = SupConLoss(
            temperature=temperature, reducer=self.reducer
        )

    def element_reduction_helper(self, losses, embeddings):
        low_condition = losses > self.low if self.low is not None else True
        high_condition = losses < self.high if self.high is not None else True
        threshold_condition = low_condition & high_condition
        num_past_filter = torch.sum(threshold_condition)
        if num_past_filter >= 1:
            loss = torch.mean(losses[threshold_condition])
        else:
            loss = torch.sum(embeddings * 0)
        return loss

    def forward(self, embeddings, targets, mask):
        losses = self.criterion(embeddings, targets)
        losses = losses["loss"]["losses"] * torch.cat((mask, mask), 0)
        return self.element_reduction_helper(losses, embeddings)


class HMLC(nn.Module):
    """
    Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework
    Paper: https://arxiv.org/abs/2204.13207
    GitHub: https://github.com/salesforce/hierarchicalContrastiveLearning
    """
    def __init__(
        self,
        temperature=0.1,
        base_temperature=1.0,
        layer_penalty=torch.exp,
        loss_type='hmc',
    ):
        super(HMLC, self).__init__()
        self.loss_type = loss_type
        self.sup_con_loss = SupConLoss(temperature)
        self.masked_sup_con_loss = MaskedSupConLoss(temperature)
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels, masks=None):
        device = torch.device("cuda"
                             ) if features.is_cuda else torch.device("cpu")

        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float("-inf"))

        for l in range(0, labels.shape[1]):
            layer_labels = labels[:, l]
            if masks == None:
                layer_loss = self.sup_con_loss(features, layer_labels)
            else:
                layer_loss = self.masked_sup_con_loss(
                    features, layer_labels, masks
                )

            if self.loss_type == "hmc":
                cumulative_loss += (
                    self.layer_penalty(
                        torch.tensor(1 /
                                     (labels.shape[1] - l)).type(torch.float)
                    ) * layer_loss
                )
            elif self.loss_type == "hce":
                layer_loss = torch.max(
                    max_loss_lower_layer.to(layer_loss.device), layer_loss
                )
                cumulative_loss += layer_loss
            elif self.loss_type == "hmce":
                layer_loss = torch.max(
                    max_loss_lower_layer.to(layer_loss.device), layer_loss
                )
                cumulative_loss += (
                    self.layer_penalty(
                        torch.tensor(1 /
                                     (labels.shape[1] - l)).type(torch.float)
                    ) * layer_loss
                )
            else:
                raise NotImplementedError("Unknown loss")

            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss
            )

        return cumulative_loss  # / labels.shape[1]
