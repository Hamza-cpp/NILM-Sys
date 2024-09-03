# Copyright 2024 OKHADIR Hamza
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F


def error(labels, outputs):
    """
    Calcualte L1 error
    """
    err = F.l1_loss(labels, outputs)
    return err


def compute_loss(
    model: Module,
    targets: Tensor,
    predictions: Tensor,
    classes: Tensor,
    class_predictions: Tensor,
) -> Tensor:
    """
    Compute the loss for model predictions.

    Args:
        model (torch.nn.Module): The model being trained.
        targets (torch.Tensor): Ground truth targets.
        predictions (torch.Tensor): Predicted values from the model.
        classes (torch.Tensor): Ground truth class labels.
        class_predictions (torch.Tensor): Predicted class labels from the model.

    Returns:
        torch.Tensor: Computed loss value.
    """
    regression_loss = F.mse_loss(predictions, targets)
    if model.classification_enabled:
        classification_loss = F.binary_cross_entropy(class_predictions, classes)
        return regression_loss + classification_loss
    return regression_loss
