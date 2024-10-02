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

from datetime import datetime
from typing import Optional, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from src.utils.metrics import error, compute_loss
from src.utils.plotting import plot_intermediate_results

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(
    epoch: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    transform: dict,
    optimizer: Optimizer,
    eval_loader: DataLoader,
    plotfilename: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Train a single epoch for a specific model and appliance.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        transform (dict): Dictionary containing mean and std for standardization.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        eval_loader (DataLoader): DataLoader for evaluation data.
        plotfilename (str, optional): Filename prefix for saving plots. Defaults to None.

    Returns:
        tuple: Mean loss and error over the epoch.
    """
    model.train()
    batch_losses, batch_errors = [], []

    start_time = datetime.now()

    for batch_index, (inputs, targets, classes) in enumerate(train_loader):
        # Prepare model input data
        inputs = torch.unsqueeze(inputs, dim=1)
        inputs, targets, classes = (
            inputs.to(device),
            targets.to(device),
            classes.to(device),
        )

        optimizer.zero_grad()
        predictions, reg_outputs, alphas, class_predictions = model(inputs)

        # Calculate prediction loss. See network architecture
        # and loss details in documentation
        loss = compute_loss(model, targets, predictions, classes, class_predictions)
        loss.backward()
        optimizer.step()

        # Calculate error
        error_value = error(targets, predictions)

        batch_losses.append(loss.cpu().item())
        batch_errors.append(error_value.cpu().item())

        if batch_index % 100 == 0:
            # Plotting sliding window samples in order to debug or
            # keep track of current testing process
            log_training_progress(epoch, batch_index, loss.cpu().item(), error_value.cpu().item())
            if plotfilename:
                plot_intermediate_results(
                    plotfilename,
                    batch_index,
                    inputs.cpu(),
                    targets.cpu(),
                    predictions.cpu(),
                    reg_outputs.cpu(),
                    alphas.cpu(),
                    class_predictions.cpu(),
                    transform,
                    model.classification_enabled,
                    loss.cpu().item(),
                    error_value.cpu().item(),
                )

    epoch_duration = (datetime.now() - start_time).seconds
    print_epoch_summary(epoch_duration)

    return np.mean(batch_losses), np.mean(batch_errors)


def eval_single_epoch(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    transform: dict,
    plotfilename: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Evaluate a single epoch for a specific model and appliance.

    Args:
        model (torch.nn.Module): The model to evaluate.
        eval_loader (DataLoader): DataLoader for the evaluation data.
        transform (dict): Dictionary containing mean and std for standardization.
        plotfilename (str, optional): Filename prefix for saving plots. Defaults to None.

    Returns:
        tuple: Mean loss and error over the evaluation epoch.
    """
    model.eval()
    batch_losses, batch_errors = [], []

    start_time = datetime.now()

    with torch.no_grad():
        for batch_index, (inputs, targets, classes) in enumerate(eval_loader):
            # Prepare model input data
            inputs = torch.unsqueeze(inputs, dim=1)
            inputs, targets, classes = (
                inputs.to(device),
                targets.to(device),
                classes.to(device),
            )

            predictions, reg_outputs, alphas, class_predictions = model(inputs)

            # Calculate loss and error
            loss = compute_loss(model, targets, predictions, classes, class_predictions)
            error_value = error(targets, predictions)

            batch_losses.append(loss.cpu().item())
            batch_errors.append(error_value.cpu().item())

            if batch_index % 100 == 0:
                # Plotting sliding window samples in order to debug or
                # keep track of current testing process
                log_evaluation_progress(batch_index, loss.cpu().item(), error_value.cpu().item())

                if plotfilename:
                    plot_intermediate_results(
                        plotfilename,
                        batch_index,
                        inputs.cpu(),
                        targets.cpu(),
                        predictions.cpu(),
                        reg_outputs.cpu(),
                        alphas.cpu(),
                        class_predictions.cpu(),
                        transform,
                        model.classification_enabled,
                        loss.cpu().item(),
                        error_value.cpu().item(),
                        eval_mode=True,
                    )

    epoch_duration = (datetime.now() - start_time).seconds
    print_epoch_summary(epoch_duration)

    return np.mean(batch_losses), np.mean(batch_errors)


def log_training_progress(
    epoch: int, batch_index: int, loss: torch.Tensor, error_value: torch.Tensor
) -> None:
    """
    Log the training progress.

    Args:
        epoch (int): Current epoch number.
        batch_index (int): Current batch index.
        loss (torch.Tensor): Current loss value.
        error_value (torch.Tensor): Current error value.
    """
    print(
        f"train epoch={epoch} batch={batch_index+1} loss={loss:.2f} err={error_value:.2f}"
    )


def log_evaluation_progress(
    batch_index: int, loss: torch.Tensor, error_value: torch.Tensor
) -> None:
    """
    Log the evaluation progress.

    Args:
        batch_index (int): Current batch index.
        loss (torch.Tensor): Current loss value.
        error_value (torch.Tensor): Current error value.
    """
    print(f"eval batch={batch_index+1} loss={loss:.2f} err={error_value:.2f}")


def print_epoch_summary(epoch_duration: int) -> None:
    """
    Print a summary of the epoch's duration.

    Args:
        epoch_duration (int): Duration of the epoch in seconds.
    """
    print("------------------------------------------")
    print(f"Epoch seconds: {epoch_duration}")
    print("------------------------------------------")
