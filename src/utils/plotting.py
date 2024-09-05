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

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(
    dataframe: pd.DataFrame, path: str, filename: str, columns: list, ylabel: str
) -> None:
    """
    Plot the selected columns of a DataFrame and save the figure to a file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the training and evaluation results.
    path : str
        The directory path where the plot image will be saved.
    filename : str
        The name of the plot image file.
    columns : list
        The columns to plot.
    ylabel : str
        The label for the y-axis of the plot.
    """
    filepath = os.path.join(path, filename)
    plt.figure(figsize=(10, 8))
    dataframe[columns].round(3).plot()
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.title(f"{ylabel} over Epochs")
    plt.savefig(filepath)
    plt.close()  # Close the plot to free up memory
    print(f"Plot saved to {filepath}")


def plot_intermediate_results(
    plotfilename: str,
    batch_index: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    reg_outputs: torch.Tensor,
    alphas: torch.Tensor,
    class_predictions: torch.Tensor,
    transform: dict,
    classification_enabled: bool,
    loss: float,
    error: float,
    eval_mode: bool = False,
) -> None:
    """
    Plot intermediate results during training or evaluation.

    Args:
        plotfilename (str): Filename prefix for saving plots.
        batch_index (int): Current batch index.
        inputs (torch.Tensor): Input data (total aggregated power).
        targets (torch.Tensor): Ground truth targets (appliance actual power).
        predictions (torch.Tensor): Model predictions.
        reg_outputs (torch.Tensor): Regression prediction branch.
        alphas (torch.Tensor): Attention weights show which parts of the input the model is focusing on to make its prediction.
        class_predictions (torch.Tensor): Classification predictions branch.
        transform (dict): Dictionary containing mean and std for standardization.
        classification_enabled (bool): Whether classification is enabled.
        loss (float): Current loss value.
        error (float): Current error value.
        eval_mode (bool, optional): Flag indicating if evaluation mode. Defaults to False.
    """
    filename_suffix = ".attention.png" if eval_mode else f".{batch_index}.png"
    filename = plotfilename + filename_suffix

    inputs, targets, predictions, reg_outputs = (
        inputs.cpu(),
        targets.cpu(),
        predictions.cpu(),
        reg_outputs.cpu(),
    )

    if transform:
        # Undo standardization for visualization
        inputs = (inputs * transform["sample_std"]) + transform["sample_mean"]
        targets = (targets * transform["target_std"]) + transform["target_mean"]
        predictions = (predictions * transform["target_std"]) + transform["target_mean"]
        reg_outputs = (reg_outputs * transform["sample_std"]) + transform["sample_mean"]
        reg_outputs /= 10.0  # Rescale regression output for visualization

    plot_time_series(
        inputs=inputs,
        targets=targets,
        predictions=predictions,
        reg_outputs=reg_outputs,
        class_predictions=class_predictions,
        alphas=alphas,
        classification_enabled=classification_enabled,
        filename=filename,
        loss=loss,
        error=error,
    )


def plot_time_series(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    reg_outputs: torch.Tensor,
    class_predictions: torch.Tensor,
    alphas: torch.Tensor,
    classification_enabled: bool,
    filename: str,
    loss: float,
    error: float,
) -> None:
    """
    Plot sliding window to visualize disaggregation results, track results in training or testing, and debug.

    Plotting multipel time series
       - Aggregated demand
       - Appliance demand
       - Disaggregation prediction
       - Regression branch prediction
       - Classification branch prediction

    Parameters
    ----------
    inputs : torch.Tensor
        Input data (total aggregated power).
    targets : torch.Tensor
        Ground truth targets (appliance actual power).
    predictions : torch.Tensor
        Model (Network) predictions.
    reg_outputs : torch.Tensor
        Regression prediction branch.
    class_predictions : torch.Tensor
        Classification predictions branch.
    alphas : torch.Tensor
        Attention weights show which parts of the input the model is focusing on to make its prediction.
    classification_enabled : bool
        Whether classification is enabled.
    filename : str
        Filename for saving the plot.
    loss : float
        Loss value.
    error : float
        Error value.
    """

    subplt_x, subplt_y = 4, 4
    plt.figure(figsize=(20, 16))
    plt.subplots_adjust(top=0.88)

    idxs = np.random.randint(len(inputs), size=(subplt_x * subplt_y))
    for i, idx in enumerate(idxs):
        input_, target_, prediction_, reg_output_, class_prediction_ = (
            inputs.detach().numpy()[idx][0],
            targets.detach().numpy()[idx],
            predictions.detach().numpy()[idx],
            reg_outputs.detach().numpy()[idx],
            class_predictions.detach().numpy()[idx],
        )
        alphas_ = alphas.detach().numpy()[idx].flatten()

        # Plot the time series data
        ax1 = plt.subplot(subplt_x, subplt_y, i + 1)
        ax2 = ax1.twinx()
        ax1.plot(
            range(len(input_)),
            input_,
            color="b",
            label="Input (total power)",
        )
        ax1.plot(
            range(len(target_)),
            target_,
            color="r",
            label="Target (appliance power)",
        )
        ax1.plot(
            range(len(reg_output_)),
            reg_output_,
            color="black",
            label="Regression prediction",
        )
        ax1.plot(
            range(len(prediction_)),
            prediction_,
            # alpha=0.5,
            color="orange",
            label="Network prediction",
        )
        ax2.fill_between(
            range(len(alphas_)),
            alphas_,
            alpha=0.5,
            color="lightgrey",
            label="Attention Weights",
        )

        if classification_enabled:
            alphas_max = np.max(alphas_)
            ax2.plot(
                range(len(class_prediction_)),
                class_prediction_ * alphas_max,
                color="cyan",
                # alpha=0.25,
                label="Classification prediction",
            )

    plt.suptitle(f"Loss: {loss:.2f} | Error: {error:.2f}")
    ax1.legend()
    ax2.legend()
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    print(f"Plot saved to {filename}")
