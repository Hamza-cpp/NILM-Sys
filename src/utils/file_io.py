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
