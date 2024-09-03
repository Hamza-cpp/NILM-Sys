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

import torch
import numpy as np
import pandas as pd
import torch.optim as optim

import src.model.model as nilmmodel
from src.data.dataset import InMemoryKoreaDataset
from src.utils.file_io import save_results_to_csv
from src.utils.plotting import plot_results
from src.utils.utils import save_model, load_model, save_dataset
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


def train_model(datapath, output, appliance, hparams, doplot=None, reload=True):
    """
    Train specific model and appliance
    """

    # Load appliance specifications and hyperparameters from
    # settings
    buildings = appliance["buildings"]["train"]
    name = appliance["name"]
    params = appliance["hparams"]
    record_err = np.inf

    # Load whether data transformation is required. See details
    # on data normalization in documentation
    transform_enabled = appliance.get("normalization", False)
    # Load specific network architecture to train
    model_type = appliance.get("model", "ModelPaper")

    # Initialize active settings described in documentation.
    # Used to identify whether an appliance is classified as active
    # Used to enableoversampling to fix sliding windows active/inactive
    # imbalance
    active_threshold = appliance.get("active_threshold", 0.15)
    active_ratio = appliance.get("active_ratio", 0.5)
    active_oversample = appliance.get("active_oversample", 2)

    transform = None  # Data transformation disabled by default

    # Load train dataset
    my_dataset = InMemoryKoreaDataset(
        datapath,
        buildings,
        name,
        windowsize=params["L"],
        active_threshold=active_threshold,
        active_ratio=active_ratio,
        active_oversample=active_oversample,
        transform_enabled=transform_enabled,
    )

    if transform_enabled:
        # Load dataset transformation parameters from dataset
        transform = {
            "sample_mean": my_dataset.sample_mean,
            "sample_std": my_dataset.sample_std,
            "target_mean": my_dataset.target_mean,
            "target_std": my_dataset.target_std,
        }
        print(transform)

    # Size train and evaluation dataset
    total_size = len(my_dataset)
    train_size = int(hparams["train_size"] * (total_size))
    eval_size = total_size - train_size

    print("============= DATASET =============")
    print(f"Total size: {total_size}".format(total_size))
    print(f"Train size: {train_size}".format(train_size))
    print(f"Eval size: {eval_size}".format(eval_size))
    print("===================================")
    print("=========== ARCHITECTURE ==========")
    pprint.pprint(appliance)
    print("===================================")

    # Split and randomize train and evaluation dataset
    train_dataset, eval_dataset = torch.utils.data.random_split(
        my_dataset, (train_size, eval_size)
    )

    # Save train dataset in order to use it in later
    # training sessions or debugging
    filename = os.path.join(output, "dataset.pt")
    save_dataset(transform, train_dataset, eval_dataset, filename)

    # Initialize train dataset loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )
    # Initialize evaluation dataset loader
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=hparams["batch_size"]
    )

    model_type = getattr(nilmmodel, model_type)
    model = model_type(params["L"], params["F"], params["K"], params["H"])
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), hparams["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    if reload:
        # Reload pretrained model in order to continue
        # previous training sessions
        filename = os.path.join(output, appliance["filename"])
        print("====================================")
        print("Reloading model: ", filename)
        print("====================================")
        transform, record_err = load_model(filename, model, optimizer)

    results = []

    start = datetime.now()
    for epoch in range(hparams["epochs"]):
        # Iterate over training epochs
        filename = os.path.join(output, appliance["filename"] + str(epoch))

        plotfilename = None
        if doplot:
            plotfilename = filename

        err_ = None
        try:
            # Train single epoch
            loss, err = train_single_epoch(
                epoch,
                model,
                train_loader,
                transform,
                optimizer,
                eval_loader,
                plotfilename,
            )
            print("==========================================")
            print(f"train epoch={epoch} loss={loss:.2f} err={err:.2f}")
            print("==========================================")

            loss_, err_ = eval_single_epoch(model, eval_loader, transform)
            print("==========================================")
            print(f"eval loss={loss_:.2f} err={err_:.2f}")
            print("==========================================")

            # tune.report(eval_loss=loss_)
            results.append([(epoch, loss, err), (epoch, loss_, err_)])

            if err_ < record_err:
                # Compare current epoch error against previous
                # epochs error (minimum historic error) to check whether current
                # trained model is better than previous ones (best historic error)
                # Set and save current trained model as best historic trained
                # model if current error is lower than historic error
                filename = os.path.join(output, appliance["filename"])
                save_model(
                    model, optimizer, hparams, appliance, transform, filename, err_
                )
                record_err = err_
        except Exception as e:
            print(e)

        scheduler.step()

    end = datetime.now()
    total_seconds = (end - start).seconds
    print("------------------------------------------")
    print(f"Total seconds: {total_seconds}")
    print("------------------------------------------")

    # Save model training results
    save_training_summary(output, results)

    return model, transform


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
