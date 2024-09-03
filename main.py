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

import argparse

import torch
from ray import tune

from src.utils.utils import load_yaml
from src.training.train import train_model
from src.testing.test import test_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_command_line_arguments() -> argparse.Namespace:
    """
    Parse command line arguments

    Supported arguments:

        --settings: Path to settings yaml file where all disaggregation scenarios
                    and model hyperparameters are described
        --appliance: Name of the appliance to train or test
        --path: Path to output folder where results are saved
        --train: Set to train or unset to test
        --tune: Set to enable automatic architecture hyperparameters tunning
        --epochs: Number of epochs to train
        --disable-plot: Disable sliding window plotting during train or test
        --disable-random: Disable randomness in processing
    """
    parser = argparse.ArgumentParser(
        description="NILM-Sys focuses on optimizing energy consumption in buildings "
        "through Non-Intrusive Load Monitoring (NILM). By leveraging advanced "
        "neural network architectures, the system disaggregates aggregate power "
        "data to estimate the power usage of individual appliances, contributing "
        "to climate change mitigation by enhancing energy efficiency in building"
    )

    parser.add_argument(
        "--settings", required=True, help="Settings yaml file path", type=str
    )
    parser.add_argument(
        "--appliance",
        required=True,
        help="Name of the appliance to train or test",
        type=str,
    )
    parser.add_argument(
        "--path", required=True, help="Output folder path", type=str                
    )
    parser.add_argument(
        "--train", action="store_true", help="Train if set, test if unset"
    )
    parser.add_argument(
        "--tune", action="store_true", help="Enable automatic hyperparameters tunning"
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to train", default=5
    )
    parser.add_argument(
        "--disable-plot",
        action="store_true",
        help="Disable sliding window plotting during train or test",
    )
    parser.add_argument(
        "--disable-random", action="store_true", help="Disable randomness in processing"
    )
    # Check for null pointer references
    if not parser.parse_args().settings:
        raise ValueError("Settings path is null")
    if not parser.parse_args().appliance:
        raise ValueError("Appliance name is null")
    if not parser.parse_args().path:
        raise ValueError("Output folder path is null")

    return parser.parse_args()


def main():
    """
    Main task called from command line. Command line arguments
    and train or test is launched
    """
    args = parse_command_line_arguments()

    if args.disable_random:  # Disable randomness
        torch.manual_seed(7)

    train = args.train
    tune_enabled = args.tune
    output = args.path
    plot_disabled = args.disable_plot

    # Load settings from YAML file where generic and appliance
    # specific details and model hyperparmeters are described
    if args.settings is None:
        raise ValueError("settings.yaml file path must be provided")

    settings = load_yaml(args.settings)
    if settings is None:
        raise ValueError("Settings file is not valid")

    appliance_name = args.appliance
    if appliance_name is None:
        raise ValueError("Appliance name must be provided")

    dataset = settings["dataset"]
    if dataset is None:
        raise ValueError("Dataset path must be provided in the setings.yaml file")

    hparams = settings["hparams"]
    if args.epochs:
        hparams["epochs"] = int(args.epochs)

    appliance = settings["appliances"].get(appliance_name)
    if appliance is None:
        raise ValueError(f"Appliance '{appliance_name}' does not exist")

    datapath = dataset["path"]
    if datapath is None:
        raise ValueError("Dataset path must be provided in the setings.yaml file")

    if train:
        # DO TRAIN

        print("==========================================")
        print("Training ONGOING")
        print("==========================================")

        if not tune_enabled:
            # If no automatic hyperparameter tunning is enabled
            # use network hyperparameter from settings and train
            # the model
            model, transform = train_model(
                datapath,
                output,
                appliance,
                hparams,
                doplot=not plot_disabled,
                reload=False,  # Do not reload models by default
            )
        else:
            # If automatic hyperparameter tunning is enabled
            # specify hyperparameters grid search and tune the model
            config = {
                "datapath": datapath,
                "output": output,
                "appliance": appliance,
                "hparams": hparams,
                "doplot": not plot_disabled,
                "reload": False,
                "tune": {
                    "F": tune.grid_search([16, 32, 64]),
                    "K": tune.grid_search([4, 8, 16]),
                    "H": tune.grid_search([256, 512, 1024]),
                },
            }
            analysis = tune.run(
                train_model_wrapper,  # Use wrapper to adapt training model
                metric="val_loss",
                mode="min",
                num_samples=5,
                config=config,
            )
            print("==========================================")
            print("Best hyperparameters")
            print(analysis.best_config)
            print("==========================================")

        print("==========================================")
        print("Training DONE")
        print("==========================================")
    else:
        # DO TEST

        print("==========================================")
        print("Testing ONGOING")
        print("==========================================")
        test_model(datapath, output, appliance, hparams, doplot=not plot_disabled)
        print("==========================================")
        print("Testing DONE")
        print("==========================================")


if __name__ == "__main__":
    main()
