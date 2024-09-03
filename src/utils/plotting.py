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
        inputs (torch.Tensor): Input data.
        targets (torch.Tensor): Ground truth targets.
        predictions (torch.Tensor): Model predictions.
        reg_outputs (torch.Tensor): Regression output from the model.
        alphas (torch.Tensor): Attention weights from the model.
        class_predictions (torch.Tensor): Class predictions from the model.
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

    plot_window(
        inputs,
        targets,
        predictions,
        reg_outputs,
        class_predictions.cpu(),
        alphas.cpu(),
        loss,
        error,
        classification_enabled,
        filename,
    )


def plot_window(
    x, y, yhat, reg, clas, alphas, loss, err, classification_enabled, filename
):
    """
    Plot sliding window to visualize disaggregation results, keep track
    of results in training or testing and debugging

    Plotting multipel time series
       - Aggregated demand
       - Appliance demand
       - Disaggregation prediction
       - Regression branch prediction
       - Classification branch prediction
    """
    subplt_x = 4
    subplt_y = 4
    plt.figure(1, figsize=(20, 16))
    plt.subplots_adjust(top=0.88)

    idxs = np.random.randint(len(x), size=(subplt_x * subplt_y))
    for i, idx in enumerate(idxs):
        x_, y_, yhat_, reg_, clas_ = (
            x.detach().numpy()[idx][0],
            y.detach().numpy()[idx],
            yhat.detach().numpy()[idx],
            reg.detach().numpy()[idx],
            clas.detach().numpy()[idx],
        )
        alphas_ = alphas.detach().numpy()[idx].flatten()
        ax1 = plt.subplot(subplt_x, subplt_y, i + 1)
        ax2 = ax1.twinx()
        ax1.plot(range(len(x_)), x_, color="b", label="x")
        ax1.plot(range(len(y_)), y_, color="r", label="y")
        ax1.plot(range(len(reg_)), reg_, color="black", label="reg")
        ax1.plot(range(len(yhat_)), yhat_, alpha=0.5, color="orange", label="yhat")
        ax2.fill_between(
            range(len(alphas_)), alphas_, alpha=0.5, color="lightgrey", label="alpha"
        )
        if classification_enabled:
            alphas_max = np.max(alphas_)
            ax2.plot(
                range(len(clas_)),
                clas_ * alphas_max,
                color="cyan",
                alpha=0.25,
                label="reg",
            )

    plt.suptitle(f"loss {loss:.2f} error {err:.2f}")
    ax1.legend()
    ax2.legend()
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
