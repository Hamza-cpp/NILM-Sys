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
import sys
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
from datetime import datetime

import src.data.redd as redd
import src.utils as utils


class Building:
    """
    Building consumption class handler - defines appliances and main
    consumption data processing.
    """

    def __init__(self, path: str, name: str, spec: dict) -> None:
        """
        Initialize Building class.

        Args:
            path (str): Path to the building data.
            name (str): Name of the building.
            spec (Dict): Specification of the building, including mains and appliances channels.
        """
        self.path = path
        self.name = name
        self.mains_channel = spec["mains"]
        self.appliances = spec["appliances"]

    @property
    def get_appliances(self) -> List[str]:
        """
        Get a list of appliance IDs (names of appliances).

        Returns:
            List[str]: List of appliance IDs.
        """
        return [appliance["id"] for appliance in self.appliances]

    def load_mains(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Load mains consumption data for a specified time interval. Using dataset specific loader

        Args:
            start_time (Optional[datetime]): Start time for loading data. Defaults to None.
            end_time (Optional[datetime]): End time for loading data. Defaults to None.

        Returns:
            pd.Series: A pandas Series object containing mains consumption data.
        """
        return redd.load(
            name="mains",
            path=self.path,
            channels=self.mains_channel,
            start_time=start_time,
            end_time=end_time,
        )

    def load_appliances(
        self,
        appliances: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load appliance consumption data for a specified time interval.

        Args:
            appliances (Optional[List[str]]): List of appliance IDs to load. Defaults to None.
            start (Optional[datetime]): Start time for loading data. Defaults to None.
            end (Optional[datetime]): End time for loading data. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing appliance consumption data.
        """
        appliances_to_load = appliances or self.appliances_ids

        df_list = [
            redd.load(
                name=appliance["id"],
                path=self.path,
                channels=appliance["channels"],
                start_time=start,
                end_time=end,
            )
            for appliance in self.appliances
            if appliance["id"] in appliances_to_load
        ]

        # WARNING: Time series inner join. Ignoring non-synced
        # datapoints from loaded chanels
        # Merge the dataframes on their indices with an inner join
        return pd.concat(df_list, axis=1, join="inner")


class NilmDataset:
    """
    NILM dataset handler for preprocessing tasks such as alignment and imputation.

    Note:
        This dataset handler is used when dataset preprocessing is required:
        - Alignment of time series data.
        - Imputation of missing data.
    """

    def __init__(self, spec_path: str, dataset_path: str) -> None:
        """
        Initialize NILM dataset handler.

        Args:
            spec_path (str): Path to the dataset specification file (YAML), contains information about the dataset structure.
            dataset_path (str): Base path where the dataset is stored.
        """
        self.path = dataset_path
        spec = utils.load_yaml(spec_path)

        dataset_full_path = os.path.join(self.path, spec["path"])
        # Load all buildings specified in settings
        self.buildings = {
            building["name"]: Building(
                os.path.join(dataset_full_path, building["path"]),
                building["name"],
                building,
            )
            for building in spec["buildings"]
        }

    def get_buildings_names(self) -> List[str]:
        """
        Get a list of building names.

        Returns:
            List[str]: List of building names.
        """
        return list(self.buildings.keys())

    def get_appliances(self, building_name: str) -> List[str]:
        """
        Get a list of appliances for a given building.

        Args:
            building_name (str): Name of the building.

        Returns:
            List[str]: List of appliance IDs.
        """
        return self.buildings[building_name].get_appliances()

    def load_mains(
        self,
        building_name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Load mains consumption data for a given building within a time interval.

        Args:
            building_name (str): Name of the building.
            start (Optional[datetime]): Start time for data loading. Defaults to None.
            end (Optional[datetime]): End time for data loading. Defaults to None.

        Returns:
            pd.Series: Mains consumption data.
        """
        return self.buildings[building_name].load_mains(start, end)

    def load_appliances(
        self,
        building_name: str,
        appliances: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load appliances consumption data for a given building within a time interval.

        Args:
            building_name (str): Name of the building.
            appliances (Optional[List[str]]): List of appliance IDs to load. Defaults to None.
            start (Optional[datetime]): Start time for data loading. Defaults to None.
            end (Optional[datetime]): End time for data loading. Defaults to None.

        Returns:
            pd.DataFrame: Appliances consumption data.
        """
        return self.buildings[building_name].load_appliances(appliances, start, end)

    @staticmethod
    def align(
        df1: pd.DataFrame, df2: pd.DataFrame, bfill: bool = False
    ) -> pd.DataFrame:
        """
        Align two timeseries with different acquisition frequencies.

        Args:
            df1 (pd.DataFrame): First DataFrame.
            df2 (pd.DataFrame): Second DataFrame.
            bfill (bool): Whether to perform backward filling for missing data. Defaults to False.

        Returns:
            pd.DataFrame: Aligned DataFrame with overlapping time indices.
        """
        # Time alignment required due to different acquisition frequencies
        if bfill:
            # Raw Backward filling for missing values
            aligned_df2 = df2.reindex(df1.index, method="bfill")
            df_aligned = pd.concat([df1, aligned_df2], axis=1, join="inner")
        else:
            df_aligned = pd.concat([df1, df2], axis=1, join="inner")

        return df_aligned.dropna()

    @staticmethod
    def impute(
        df: pd.DataFrame, gapsize: int = 20, subseqsize: int = 28800
    ) -> List[pd.DataFrame]:
        """
        Impute missing data in time series and split into valid subsequences.

        Data preprocessing to impute small gaps and ignore larg gaps
        Ignore non 100% coverage days

        Extract from "Subtask Gated Networks for Non-Intrusive Load Monitoring" paper

        For REDD dataset,we preprocessed with the following procedure
        to handle missing values. First, we split the sequence so that the
        duration of missing values in subsequence is less than 20 seconds.
        Second,we filled the  missing values in each subsequence by
        thebackward filling method. Finally, we only used the subsequences
        with more than one-day duration

        Args:
            df (pd.DataFrame): Input DataFrame to preprocess.
            gapsize (int): Maximum size of gaps to fill. Defaults to 20.
            subseqsize (int): Minimum size of subsequences to retain. Defaults to 28800.

        Returns:
            List[pd.DataFrame]: List of preprocessed DataFrames with imputed data.
        """
        df = df.sort_index()
        start, end = df.index[0], df.index[-1]

        # Appliance time series are not aligned to 3s (ie. 3,4 sec period)
        # Use 1sec reindex in order to align to 3sec timeserie and fill gaps
        df = df.reindex(pd.date_range(start, end, freq="1S"), method="bfill", limit=4)
        df = df.reindex(pd.date_range(start, end, freq="3S")).dropna(how="any")
        df.fillna(method="bfill", limit=gapsize, inplace=True)

        # Identify and split subsequences based on gap sizes
        df["rowindex"] = range(df.shape[0])
        df = df[df.iloc[:, 0].notna()]

        diffsec = df.index.to_series().diff().dt.total_seconds()
        # Find big gaps to split data in subsequences
        gaps = diffsec > gapsize

        subsequences = []
        start = df.index[0]

        for idx, (gap_idx, gap_size) in enumerate(zip(df.index[gaps], diffsec[gaps])):
            end = gap_idx - pd.Timedelta(seconds=gap_size)
            subseq = df[start:end]
            if subseq.shape[0] > subseqsize:
                subsequences.append(subseq.drop(columns=["rowindex"]))
            start = gap_idx

        # Include the last subsequence
        last_subseq = df[start : df.index[-1]]
        if last_subseq.shape[0] > subseqsize:
            subsequences.append(last_subseq.drop(columns=["rowindex"]))

        return subsequences

    def load(
        self,
        building_name: str,
        appliances: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        bfill: bool = False,
    ) -> List[pd.DataFrame]:
        """
        Load and preprocess data for a given building and appliances.

        Args:
            building_name (str): Name of the building.
            appliances (Optional[List[str]]): List of appliance IDs to load. Defaults to None.
            start (Optional[datetime]): Start time for data loading. Defaults to None.
            end (Optional[datetime]): End time for data loading. Defaults to None.
            bfill (bool): Whether to perform backward filling for missing data. Defaults to False.

        Returns:
            List[pd.DataFrame]: List of preprocessed DataFrames.
        """
        aligned_data = self.align(
            self.load_mains(building_name, start, end),
            self.load_appliances(building_name, appliances, start, end),
            bfill,
        )
        return self.impute(aligned_data)

    def load_raw(
        self,
        building_name: str,
        appliances: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        bfill: bool = False,
    ) -> pd.DataFrame:
        """
        Load raw (unimputed) data for a given building and appliances.

        Args:
            building_name (str): Name of the building.
            appliances (Optional[List[str]]): List of appliance IDs to load. Defaults to None.
            start (Optional[datetime]): Start time for data loading. Defaults to None.
            end (Optional[datetime]): End time for data loading. Defaults to None.
            bfill (bool): Whether to perform backward filling for missing data. Defaults to False.

        Returns:
            pd.DataFrame: Raw aligned data.
        """
        return self.align(
            self.load_mains(building_name, start, end),
            self.load_appliances(building_name, appliances, start, end),
            bfill,
        )


class InMemoryDataset(Dataset):
    """
    In-memory dataset for NILM (Non-Intrusive Load Monitoring) tasks.

    WARNING: This dataset is stored entirely in memory, which may lead to
    potential memory overrun issues depending on the size of the dataset.
    Not used in current analysis due to the availability of preprocessed
    datasets (non-public available and obtained during the ongoing project).
    """

    def __init__(
        self,
        spec: str,
        path: str,
        buildings: List[str],
        appliance: str,
        windowsize: int = 34459,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> None:
        """
        Initializes the InMemoryDataset.

        Parameters
        ----------
        spec : str
            Path to the dataset specification file.
        path : str
            Path to the root directory of the dataset.
        buildings : List[str]
            List of building names to include in the dataset.
        appliance : str
            The appliance ID for which the data is being collected.
        windowsize : int, optional
            Size of the sliding window, by default 34459.
        start : pd.Timestamp, optional
            Start time for filtering data, by default None.
        end : pd.Timestamp, optional
            End time for filtering data, by default None.
        """
        super().__init__()
        self.buildings = buildings
        self.appliance = appliance
        self.windowsize = windowsize

        dataset = NilmDataset(spec, path)

        # Dataset is structured as multiple long size windows
        # As sliding windows are used to acces data, a lookup-table
        # is created as sequential index to reference each sliding
        # window (long window + offset within long window).
        self.data = []  # List to store the data of all buildings
        self.datamap = {}  # Lookup table for sliding window indices

        data_index = 0
        window_index = 0

        for building in buildings:
            for long_window in dataset.load(building, [appliance], start, end):
                # Calculate number of sliding windows in the long time window
                n_windows = long_window.shape[0] - windowsize + 1

                # Add loaded data to dataset
                self.data.append(long_window.reset_index())

                # Update data index iteraring over all sliding windows in
                # dataset. Each of the indexes in global map corresponds
                # to specific long time window and offset
                self.datamap.update(
                    {window_index + i: (data_index, i) for i in range(n_windows)}
                )

                data_index += 1
                window_index += n_windows

        self.total_size = window_index

    def __len__(self) -> int:
        """
        Returns the total number of samples (sliding windows) in the dataset.
        """
        return self.total_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of (sample, target) for the given index.

        Parameters
        ----------
        idx : int
            Index of the sliding window sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the input features (mains consumption) and
            the target output (appliance consumption) as PyTorch tensors.
        """
        # Each of the indexes in global map corresponds
        # to specific long time window and offset. Obtain
        # long time window and offset
        data_index, window_index = self.datamap[idx]

        # Obtain start end offset in the long time window
        start = window_index
        end = self.windowsize + window_index

        # Extract the input (mains) and target (appliance) data
        sample = self.data[data_index].loc[start:end, "mains"]
        target = self.data[data_index].loc[start:end, self.appliance]

        return torch.tensor(sample.values, dtype=torch.float32), torch.tensor(
            target.values, dtype=torch.float32
        )


class InMemoryKoreaDataset(Dataset):
    """
    Inmemory dataset
    WARNING: Not the best option, due potential memory overrun but did not fail

    Arguments:
        windowsize: Sliding window size
        active_threshold: Active threshold used in classification
           Default value in paper 15W
        active_ratio: In order to prevent imbalance in data it's required
           to balance number of active/inactive appliance windows. In most
           of the cases the number of inactive windows is larger than
           the number of active windows. Active ratio forces the ratio
           between active/inactive windows by removing active/inactive
           windows (in most cases inactive windows) till fulfilling the ratio
        active_oversample: In order to prevent overfitting oversampling is done
        in active windows. This argument forces random oversampling
        active_oversample times available active windows
        transform_enabled: Used to enable data preprocessing transformation,
           in this case standardization
        transform: Transformation properties, in case of standardization
           mean and standard deviation
    """

    sample_mean = None
    sample_std = None
    target_mean = None
    target_std = None

    def __init__(
        self,
        path,
        buildings,
        appliance,
        windowsize=496,
        active_threshold=15.0,
        active_ratio=None,
        active_oversample=None,
        transform_enabled=False,
        transform=None,
    ):
        super().__init__()

        self.transform_enabled = transform_enabled

        self.appliance = appliance
        self.windowsize = windowsize
        self.active_threshold = active_threshold

        # Dataset is structured as multiple long size windows
        self.data = []
        # As sliding windows are used to acces data, a lookup-table
        # is created as sequential index to reference each sliding
        # window (long window + offset within long window).
        self.datamap = {}

        filenames = os.listdir(path)

        columns = ["main", self.appliance]

        # Using original long time windows as non-related time interval windows
        # in order to prevent mixing days and concatenating not continuous
        # data. Original data has gaps between dataset files
        self.data = [
            pd.read_csv(os.path.join(path, filename), usecols=columns, sep=",")
            for filename in filenames
            for building in buildings
            if filename.startswith(building)
        ]

        df = pd.concat(self.data)
        # Data transformation
        if transform_enabled:
            if transform:
                self.sample_mean = transform["sample_mean"]
                self.sample_std = transform["sample_std"]
                self.target_mean = transform["target_mean"]
                self.target_std = transform["target_std"]
            else:
                self.sample_mean = df["main"].mean()
                self.sample_std = df["main"].std()
                self.target_mean = df[appliance].mean()
                self.target_std = df[appliance].std()

        data_index = 0
        window_index = 0

        for subseq in self.data:
            n_windows = subseq.shape[0] - windowsize + 1  # +1 why?
            # Update data index iteraring over all sliding windows in
            # dataset. Each of the indexes in global map corresponds
            # to specific long time window and offset
            self.datamap.update(
                {window_index + i: (data_index, i) for i in range(n_windows)}
            )
            data_index += 1
            window_index += n_windows

        self.total_size = window_index

        if active_ratio:
            # Fix imbalance required
            map_indexes = list(self.datamap.keys())
            # Shuffle indexes in order to prevent oversampling using same
            # building or continuous windows
            random.shuffle(map_indexes)

            # Active and inactive buffers are used to manage classified
            # sliding windows and use them later to fix imbalance
            active_indexes = []
            inactive_indexes = []

            # Classify every sliding window as active or inactive using
            # active_threshold as threshold
            for i, index in enumerate(map_indexes):
                data_index, window_index = self.datamap[index]
                start = window_index
                end = self.windowsize + window_index

                # Retreive sliding window from data
                subseq = self.data[data_index].loc[start : (end - 1), self.appliance]
                if subseq.shape[0] != self.windowsize:
                    continue

                # Fill active and inactive buffers to be used later to
                # fix imbalance
                if (subseq > active_threshold).any():  # is there any active ?
                    active_indexes.append(index)
                else:
                    inactive_indexes.append(index)

                if (i % 1000) == 0:
                    print(
                        "Loading {}: {}/{}".format(self.appliance, i, len(map_indexes))
                    )
            if active_oversample:
                # If oversample is required increase representation
                active_indexes = active_indexes * active_oversample

            # Identify imbalance by calculating active/inactive ratio
            n_active = len(active_indexes)
            n_inactive = len(inactive_indexes)

            # Update number of active/inactive windows to fulfill required
            # ratio and fix imbalance
            n_inactive_ = int((n_active * (1.0 - active_ratio)) / active_ratio)
            n_active_ = int((n_inactive * active_ratio) / (1.0 - active_ratio))

            if n_inactive > n_inactive_:
                n_inactive = n_inactive_
            else:
                n_active = n_active_

            # Obtain valid indexes after imbalance analysis
            valid_indexes = active_indexes[:n_active] + inactive_indexes[:n_inactive]

            # Update datamap with fixed indexes in order to point to
            # proper sliding windows
            datamap = {}
            for dst_index, src_index in enumerate(valid_indexes):
                datamap[dst_index] = self.datamap[src_index]
            self.datamap = datamap
            self.total_size = len(self.datamap.keys())

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Loader asking for specific sliding window in specific index
        # Calculate long time window and offset in order to retrieve data
        # Input data is obtained from mains time serie, target data is
        # obtained from appliance timeserie and classification is
        # done over mains time serie
        data_index, window_index = self.datamap[idx]
        start = window_index
        end = self.windowsize + window_index

        # Retreive mains data as sample data
        sample = self.data[data_index].loc[start : (end - 1), "main"]
        # Retreive appliance data as target data
        target = self.data[data_index].loc[start : (end - 1), self.appliance]

        # Calculate classification
        classification = torch.zeros(target.values.shape[0])
        if self.active_threshold:
            classification = (target.values > self.active_threshold).astype(int)

        # WARNING: This is not the proper way as both train and test values
        # used. It's just a first approach
        if self.transform_enabled:
            # Standarization enabled
            sample = (sample - self.sample_mean) / self.sample_std
            target = (target - self.target_mean) / self.target_std

        return (
            torch.tensor(sample.values, dtype=torch.float32),  # Input
            torch.tensor(target.values, dtype=torch.float32),  # Target
            torch.tensor(classification, dtype=torch.float32),  # Classification
        )


if __name__ == "__main__":
    # Default dataset handler used to explore data in colab
    # not used in training or prediction

    spec = sys.argv[1]
    path = sys.argv[2]
    appliance = sys.argv[3]

    # NOTE: Raw dataset explorer
    # from datetime import datetime
    # import pytz
    # tz = pytz.timezone("US/Eastern")
    # start = datetime(2011, 4, 20, 0,0,0)
    # end = datetime(2011, 4, 22, 0,0,0)
    # start = tz.localize(start)
    # end = tz.localize(end)

    # building = "building1"
    # appliances = ["refrigerator"]
    # dataset = NilmDataset(spec, path)
    # raw_mains = dataset.load_mains(building)
    # raw_appliances = dataset.load_appliances(building, appliances)

    # raw_df = dataset.load_raw(building, appliances)
    # clean_df = dataset.load(building, appliances)

    # buildings = ["building1", "building2"]
    # my_dataset = InMemoryDataset(spec, path, buildings, "refrigerator")

    # NOTE: Korea dataset explorer
    buildings = ["redd_house1"]
    my_dataset = InMemoryKoreaDataset(path, buildings, appliance)
