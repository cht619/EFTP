import torch
import numpy as np
from torch import nn, optim
from torchsummary import summary
from copy import deepcopy
from mmengine.structures import PixelData
import torch.nn.functional as F
from ..normalization.ema import TeaStu_base, create_ema_model
from .sourceonly import SourceOnly
from .prompt_utils import DenseVisualPrompt, SparseVisualPrompt, position_by_entropy, position_by_uncertainty
from mmseg.models.utils import resize
from .JSD_loss import calc_jsd_multiscale


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # x should be [b, c, h, w]
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
