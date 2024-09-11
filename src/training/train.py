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
import pprint
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import src.model.model as nilmmodel
from src.data.dataset import InMemoryKoreaDataset
from src.utils.plotting import plot_results
from src.utils.file_io import save_results_to_csv, save_model, load_model, save_dataset
from src.training.train_eval_epoch import train_single_epoch, eval_single_epoch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def save_training_summary(output_path: str, results: list) -> None:
    """
    Save training summary results to CSV files and plot graphs for analysis.

    This function generates CSV files and plots comparing training and evaluation loss and error,
    which can help diagnose underfitting, overfitting, or good fitting.

    Parameters
    ----------
    output_path : str
        The directory path where the results and plots will be saved.
    results : list
        A list of tuples containing epoch number, training loss, training error, evaluation loss, and evaluation error.
    """
    # Create a DataFrame from the results list
    df = pd.DataFrame(
        [
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_err": train_err,
                "eval_loss": eval_loss,
                "eval_err": eval_err,
            }
            for ((epoch, train_loss, train_err), (_, eval_loss, eval_err)) in results
        ]
    ).set_index("epoch")

    # Define columns for loss and error
    loss_columns = ["train_loss", "eval_loss"]
    error_columns = ["train_err", "eval_err"]

    # Save results to CSV
    save_results_to_csv(df, output_path, "results-loss.csv", loss_columns)
    save_results_to_csv(df, output_path, "results-error.csv", error_columns)

    # Plot results
    plot_results(df, output_path, "results-loss.png", loss_columns, "Loss")
    plot_results(df, output_path, "results-error.png", error_columns, "Error")


def train_model(
    datapath: str,
    output: str,
    appliance: dict,
    hparams: dict,
    doplot: Optional[bool] = None,
    reload: bool = True,
) -> Tuple[Any, Optional[Dict[str, float]]]:
    """
    Trains a specific model for a given appliance using the provided hyperparameters.

    Parameters
    ----------
    datapath : str
        Path to the dataset.
    output : str
        Output path for saving the model, dataset, and results.
    appliance : dict
        Dictionary containing appliance specifications and hyperparameters.
    hparams : dict
        Dictionary containing hyperparameters for training (e.g., epochs, learning rate).
    doplot : bool, optional
        Whether to plot intermediate results during training, by default None.
    reload : bool, optional
        Whether to reload the pre-trained model and continue training, by default True.

    Returns
    -------
    Tuple[model, Optional[dict]]
        The trained model and a dictionary containing transformation parameters (e.g, mean and standard deviation...).
    """

    model, transform, train_loader, eval_loader = initialize_training_and_data_loaders(
        appliance, datapath, hparams
    )

    optimizer, scheduler = setup_optimizer(model, hparams)

    if reload:
        # Reload pretrained model in order to continue
        # previous training sessions
        filename = os.path.join(output, appliance["filename"])
        print("====================================")
        print("Reloading model: ", filename)
        print("====================================")
        transform, record_err = load_model(filename, model, optimizer)
    else:
        record_err = np.inf

    results = train_epochs(
        model,
        optimizer,
        scheduler,
        train_loader,
        eval_loader,
        output,
        appliance,
        hparams,
        doplot,
        transform,
        record_err,
    )

    return model, transform


def initialize_training_and_data_loaders(
    appliance: Dict[str, Any], datapath: str, hparams: Dict[str, Any]
) -> Tuple[nn.Module, Optional[Dict[str, float]], DataLoader, DataLoader]:
    """
    Initializes the model, dataset settings from (Settings Yaml), and prepares data loaders.

    This function takes in the appliance configuration, dataset path, and hyperparameters from the
    settings YAML file. It then initializes the model architecture, dataset, and data loaders for
    training and evaluation. The function also saves the dataset for future use.

    Parameters
    ----------
    appliance : Dict[str, Any]
        Dictionary containing appliance specifications and hyperparameters.
    datapath : str
        Path to the dataset.
    hparams : Dict[str, Any]
        Hyperparameters for training.

    Returns
    -------
    model : nn.Module
        Initialized model.
    transform : Optional[Dict[str, float]]
        Transformation dictionary.
    train_loader : DataLoader
        Data loader for the training dataset.
    eval_loader : DataLoader
        Data loader for the evaluation dataset.
    """

    # Load hyperparameters from the settings
    params = appliance["hparams"]

    # Load dataset and transformation settings
    transform_enabled = appliance.get("normalization", False)
    dataset = load_dataset(appliance, datapath, params, transform_enabled)

    # Initialize transformation settings
    # The transformation settings are used to normalize the data
    transform = None
    if transform_enabled:
        transform = {
            "sample_mean": dataset.sample_mean,
            "sample_std": dataset.sample_std,
            "target_mean": dataset.target_mean,
            "target_std": dataset.target_std,
        }

    total_size = len(dataset)
    train_size = int(hparams["train_size"] * total_size)
    eval_size = total_size - train_size

    print("============= DATASET =============")
    print(f"Total size:     {total_size}")
    print(f"Train size:     {train_size}")
    print(f"Eval size:      {eval_size}")
    print("===================================")

    # Split the dataset into a training set and an evaluation set
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Save the dataset for future use
    # The dataset is saved in a file called "dataset.pt"
    save_dataset(
        transform, train_dataset, eval_dataset, os.path.join(datapath, "dataset.pt")
    )

    print("=========== ARCHITECTURE ==========")
    pprint.pprint(appliance)
    if transform_enabled:
        print("=========== TRANSFORMATION ==========")
        pprint.pprint(transform)
    print("===================================")

    # Create data loaders
    # The data loaders are used to load the data in batches
    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=hparams["batch_size"], shuffle=False
    )

    # Initialize the model architecture
    model = initialize_model(appliance, params)

    return model, transform, train_loader, eval_loader


def load_dataset(
    appliance: dict, datapath: str, params: dict, transform_enabled: bool
) -> Any:
    """
    Loads the dataset for the given appliance.

    Parameters
    ----------
    appliance : dict
        Appliance configuration.
    datapath : str
        Path to the dataset.
    params : dict
        Hyperparameters for the dataset loading.
    transform_enabled : bool
        Whether to enable data transformation.

    Returns
    -------
    Dataset
        Loaded dataset.
    """

    buildings = appliance["buildings"]["train"]
    dataset = InMemoryKoreaDataset(
        datapath,
        buildings,
        appliance["name"],
        windowsize=params["L"],
        active_threshold=appliance.get("active_threshold", 0.15),
        active_ratio=appliance.get("active_ratio", 0.5),
        active_oversample=appliance.get("active_oversample", 2),
        transform_enabled=transform_enabled,
    )
    return dataset


def initialize_model(appliance: dict, params: dict) -> torch.nn.Module:
    """
    Initializes the model for training based on the appliance configuration.

    Parameters
    ----------
    appliance : dict
        Appliance configuration.
    params : dict
        Hyperparameters for the model.

    Returns
    -------
    torch.nn.Module
        Initialized model.
    """
    # using getattr() to dynamically access a class from the nilmmodel module
    # based on the value obtained from appliance.get("model", "ModelPaper") return.
    model_class = getattr(nilmmodel, appliance.get("model", "ModelPaper"))
    model = model_class(params["L"], params["F"], params["K"], params["H"])
    model = model.to(device)
    return model


def setup_optimizer(
    model: nn.Module, hparams: dict
) -> Tuple[optim.Optimizer, optim.lr_scheduler.StepLR]:
    """
    Set up the optimizer and learning rate scheduler.

    The optimizer is responsible for updating the model's parameters to
    minimize the loss. The learning rate scheduler is responsible for
    adjusting the learning rate over time.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    hparams : dict
        Hyperparameters for training.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The optimizer. The optimizer is responsible for updating the
        model's parameters to minimize the loss.
    scheduler : torch.optim.lr_scheduler.StepLR
        The learning rate scheduler.
        the current learning rate by gamma.
    """

    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])
    # Set up the learning rate scheduler
    # The learning rate is adjusted every step_size epochs by multiplying
    # the current learning rate by gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    return optimizer, scheduler


def train_epochs(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.StepLR,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    output: str,
    appliance: dict,
    hparams: dict,
    doplot: Optional[bool],
    transform: Optional[Dict[str, float]],
    record_err: float,
) -> List[Tuple[Tuple[int, float, float], Tuple[int, float, float]]]:
    """
    Trains the model over the specified number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    optimizer : optim.Optimizer
        The optimizer for training.
    scheduler : optim.lr_scheduler.StepLR
        The learning rate scheduler.
    train_loader : DataLoader
        DataLoader for the training dataset.
    eval_loader : DataLoader
        DataLoader for the evaluation dataset.
    output : str
        Output path for saving models and results.
    appliance : dict
        Appliance details.
    hparams : dict
        Hyperparameters.
    doplot : bool
        Whether to plot results.
    transform : dict
        Data transformation parameters.
    record_err : float
        Best recorded error for the model.

    Returns
    -------
    List[Tuple[Tuple[int, float, float], Tuple[int, float, float]]]
        Results from each epoch containing loss and error.
    """
    results = []
    start = datetime.now()

    for epoch in range(hparams["epochs"]):
        filename = os.path.join(output, f"{appliance['filename']}_{epoch}")
        plotfilename = filename if doplot else None

        try:
            # Training and evaluation for each epoch
            train_loss, train_err = train_single_epoch(
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                transform=transform,
                optimizer=optimizer,
                eval_loader=eval_loader,
                plotfilename=plotfilename,
            )
            eval_loss, eval_err = eval_single_epoch(
                model=model,
                eval_loader=eval_loader,
                transform=transform,
                plotfilename=plotfilename,
            )

            results.append(
                [(epoch, train_loss, train_err), (epoch, eval_loss, eval_err)]
            )

            print(f"=================== Epoch {epoch} SUMMARY =======================")
            print(f"Train Loss={train_loss:.2f}, Train Err={train_err:.2f}")
            print(f"Eval Loss={eval_loss:.2f}, Eval Err={eval_err:.2f}")
            print("=================================================================")

            # Save the best model
            if eval_err < record_err:
                save_model(
                    model=model,
                    optimizer=optimizer,
                    hparams=hparams,
                    appliance=appliance,
                    transform=transform,
                    file_name_model=os.path.join(output, appliance["filename"]),
                    error=eval_err,
                )
                record_err = eval_err

        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")

        scheduler.step()

    end = datetime.now()
    duration = end - start
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n\n********************************************")
    print(f"Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("********************************************\n\n")

    save_training_summary(output, results)

    return results


def train_model_wrapper(config):
    """
    Wrapper to adapt model training to tune interface
    """
    datapath = config["datapath"]
    output = config["output"]
    appliance = config["appliance"]
    hparams = config["hparams"]
    doplot = config["doplot"]
    reload = config["reload"]
    tune_hparams = config["tune"]

    appliance["hparams"]["F"] = tune_hparams["F"]
    appliance["hparams"]["K"] = tune_hparams["K"]
    appliance["hparams"]["H"] = tune_hparams["H"]

    return train_model(datapath, output, appliance, hparams, doplot, reload)
