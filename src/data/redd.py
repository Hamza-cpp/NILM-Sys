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
from typing import List, Optional
from datetime import datetime

# Acquisition properties
#   Timezone: US/Eastern
#   Frequency: 1 Hz

CHANNEL_NAME_TEMPLATE = "channel_{}.dat"
TIMEZONE = "US/Eastern"


def load(
    name: str,
    path: str,
    channels: List[int],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> pd.Series:
    """
    Load and preprocess REDD dataset files, merging multiple channels and
    filtering based on a specified time range.

    Args:
        name (str): The name to assign to the resulting series.
        path (str): The directory path where the dataset files are located.
        channels (List[int]): A list of channel numbers to load.
        start_time (Optional[datetime]): Start time for filtering the data.
        end_time (Optional[datetime]): End time for filtering the data.

    Returns:
        pd.Series: A pandas Series object with the combined and filtered data.
    """

    # Read and concatenate data from specified channels
    data_frames = []
    for channel in channels:
        file_path = os.path.join(path, CHANNEL_NAME_TEMPLATE.format(channel))
        df = pd.read_csv(
            file_path, sep=" ", names=["timestamp", name], index_col="timestamp"
        )
        data_frames.append(df)

    # Merge all dataframes on their timestamp indices with inner join
    combined_df = pd.concat(data_frames, axis=1, join="inner")

    # Sum across the channels for each timestamp
    summed_series = combined_df.sum(axis=1)

    # Convert timestamps to datetime in the specified timezone
    summed_series.index = pd.to_datetime(
        summed_series.index, unit="s", utc=True
    ).tz_convert(TIMEZONE)

    # Remove duplicate timestamps, keeping the first occurrence
    summed_series = summed_series[~summed_series.index.duplicated(keep="first")]

    # Filter the series by the provided time range, if specified
    if start_time or end_time:
        start_time = (
            pd.to_datetime(start_time) if start_time else summed_series.index[0]
        )
        end_time = pd.to_datetime(end_time) if end_time else summed_series.index[-1]
        summed_series = summed_series[start_time:end_time]

    # Set the name for the resulting series
    summed_series.name = name

    return summed_series.sort_index()
