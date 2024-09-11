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
import yaml
import pprint

import torch
import pandas as pd


def save_results_to_csv(
    dataframe: pd.DataFrame, path: str, filename: str, columns: list
) -> None:
    """
    Save selected columns of a DataFrame to a CSV file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the training and evaluation results.
    path : str
        The directory path where the CSV file will be saved.
    filename : str
        The name of the CSV file.
    columns : list
        The columns to save in the CSV file.
    """
    filepath = os.path.join(path, filename)
    dataframe[columns].round(3).to_csv(filepath, sep=";")
    print(f"Results saved to {filepath}")


def load_yaml(path):
    """
    Load YAML file
    """
    _yaml = yaml.safe_load(open(path))
    return _yaml if _yaml else {}


def save_model(model, optimizer, hparams, appliance, transform, file_name_model, error):
    """
    Save the model, optimizer, hyperparameters, appliance configuration, and
    preprocessing transform to a file using PyTorch's torch.save().
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hparams": hparams,
            "appliance": appliance,
            "transform": transform,
            "error": error,
        },
        file_name_model,
    )


def load_model(file_name_model, model, optimizer=None):
    """
    Load model and metadata from file
    """
    if torch.cuda.is_available():
        state = torch.load(file_name_model)
    else:
        state = torch.load(file_name_model, map_location=torch.device("cpu"))

    model.load_state_dict(state["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    hparams = state.get("hparams", None)
    appliance = state.get("appliance", None)

    transform = state.get("transform", None)
    error = state.get("error", None)

    print("=========== ARCHITECTURE ==========")
    print("Reloading appliance")
    pprint.pprint(appliance)
    print("Reloading transform")
    pprint.pprint(transform)
    print("===================================")
    return transform, error


def save_dataset(transform, train_, test_, filename):
    """
    Save training and testing dataset to file
    """
    torch.save({"transform": transform, "train": train_, "test": test_}, filename)
