import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================
# Label Smoothing
# ====================================================


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim

    def forward(self, input, target):
        pred = input.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# ====================================================
# FocalCosineLoss
# ====================================================
class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=0.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(
            input,
            F.one_hot(target, num_classes=input.size(-1)),
            self.y,
            reduction=reduction,
        )

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


# https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/215910
# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075s
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):

        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss