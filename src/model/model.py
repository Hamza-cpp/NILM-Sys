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

"""
Neural Network Architectures for Appliance Disaggregation
=========================================================

inspired by the paper
`Subtask Gated Networks for Non-Intrusive Load Monitoring`.
-----------------------------------------------------------

This module provides a collection of neural network architectures designed for appliance disaggregation tasks. Each architecture is implemented as a PyTorch model and can be easily extended or modified for specific use cases.

Models
------

The following models are available in this module:
1. **ModelPaper**
   This model implements the architecture described in the paper "Subtask Gated Networks for Non-Intrusive Load Monitoring". It features both regression and classification branches, with attention mechanisms in the regression branch.

2. **ModelPaperBackward**
   This model is a backward-compatible version of `ModelPaper`, designed to match the architecture of previously trained models.

3. **ModelOnlyRegression**
   This model implements only the regression branch of the `ModelPaper` architecture, removing the classification branch.

4. **ModelClassAttention**
   This model extends the `ModelPaper` architecture by incorporating attention mechanisms in both the regression and classification branches.

Usage
-----
To use these models, simply import the desired model class and instantiate it with the required parameters. For example:

    >>> from this_module import ModelPaper
    >>> model = ModelPaper(sequence_length=100, num_filters=32, kernel_size=5, hidden_units=128)
You can then use the model instance for training, evaluation, or inference.

License
-------
This module is licensed under [the Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). See the ***LICENSE*** file for details.
"""

import torch
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Attention mechanism for models. Adaptable to various input dimensions.

    Attributes
    ----------
    dim : int
        Dimensionality of the input features.
    W : nn.Linear
        Linear layer to transform the input.
    V : nn.Linear
        Linear layer to compute the attention scores.
    """

    def __init__(self, dim: int = 5) -> None:
        """
        Initializes the AdditiveAttention module.

        Parameters
        ----------
        dim : int, optional
            Dimensionality of the input features to the attention mechanism, by default 5.
        """
        super().__init__()
        self.dim = dim

        # Linear transformations for additive attention mechanism
        self.W = nn.Linear(self.dim, self.dim)
        self.V = nn.Linear(self.dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the additive attention mechanism.

        Parameters
        ----------
        h : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns
        -------
        output : torch.Tensor
            Output tensor (the context vector) after applying attention, shape (batch_size, input_dim).
        alphas : torch.Tensor
            Attention weights for each input, shape (batch_size, sequence_length, 1).
        """
        # Paper attenation mechanism
        #   et = V*tanh(W*ht + b)
        #   αt = softmax(et)
        #   c = sum(αt*ht)

        # Compute the linear transformation and apply tanh activation
        layer_1 = torch.tanh(self.W(h))

        # Compute attention scores using V
        layer_2 = self.V(layer_1)

        # Compute the attention weights using softmax
        alphas = F.softmax(layer_2, dim=1)

        # Apply attention weights to input
        # Element-wise multiplication, broadcasting alphas to match h's shape
        c = h * alphas  # [batch, l, 2*h] x [batch, l, 1] = [batch, l, 2*h]

        # Sum over the sequence length dimension to compute the context vector
        output = torch.sum(
            c, dim=1
        )  # sum elements in dimension 1 (seq_length) [batch, 2*h]

        return output, alphas


class AdditiveAttentionBackwards(AdditiveAttention):
    """
    Backward-compatible version of Additive Attention for legacy models.

    This class maintains compatibility with older models by preserving the
    forward method signature and output.

    NOTE
    ----
        Nearly same implementation as main additive attention but used in order to make
        it backwards compatible as some models have already been trained using this mode.
        Otherwise fails loading the model due non-matching architecture
    """

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for backward-compatible additive attention.

        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim)

        Returns:
            output (torch.Tensor): The attended context vector of shape (batch_size, dim)
        """
        output, _ = super().forward(h)
        return output


class ModelPaper(nn.Module):
    """
    Neural network architecture for appliance disaggregation inspired by
    the paper "Subtask Gated Networks for Non-Intrusive Load Monitoring".

    This model performs both regression and classification tasks
    using convolutional, LSTM, and attention mechanisms.

    Attributes
    ----------
    regression_enabled : bool
        Flag to enable the regression branch of the model.
    classification_enabled : bool
        Flag to enable the classification branch of the model.
    convolutional_layers : nn.Sequential
        Convolutional layers for feature extraction.
    lstm_layer : nn.LSTM
        Bidirectional LSTM layer for sequential data processing.
    attention_layer : AdditiveAttention
        Attention layer for focusing on important features.
    regression_layers : nn.Sequential
        Fully connected layers for regression output.
    classification_conv_layers : nn.Sequential
        Convolutional layers for classification feature extraction.
    flatten_layer : nn.Flatten
        Flatten layer to prepare features for the fully connected layers.
    classification_fc_layers : nn.Sequential
        Fully connected layers for classification output.
    """

    def __init__(
        self,
        sequence_length: int,
        num_filters: int,
        kernel_size: int,
        hidden_units: int,
    ) -> None:
        """
        Initializes the ModelPaper architecture.

        Parameters
        ----------
        sequence_length : int
            Length of the output sequence.
        num_filters : int
            Number of filters for convolutional layers.
        kernel_size : int
            Kernel size for convolutional layers.
        hidden_units : int
            Number of hidden units for the LSTM.
        """
        super().__init__()

        self.regression_enabled = True
        self.classification_enabled = True

        # Convolutional layers for regression path
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
        )

        # LSTM layer for sequential data
        self.lstm_layer = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Attention mechanism
        self.attention_layer = AdditiveAttention(dim=(2 * hidden_units))

        # Fully connected layers for regression output
        self.regression_layers = nn.Sequential(
            nn.Linear(2 * hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, sequence_length),
        )

        # Convolutional layers for classification path
        self.classification_conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=30, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
        )  # Output shape: [batch, 50, sequence_length - 33]

        # Flatten layer for classification
        self.flatten_layer = nn.Flatten(
            start_dim=1
        )  # flatten shape: [batch, (sequence_length - 33)*50]

        # Fully connected layers for classification output
        self.classification_fc_layers = nn.Sequential(
            nn.Linear((sequence_length - 33) * 50, 1024),
            nn.ReLU(),
            nn.Linear(1024, sequence_length),
            nn.Sigmoid(),
        )

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).

        Returns
        -------
        combined_output : torch.Tensor
            Combined (*) output of regression and classification paths.
        regression_output : torch.Tensor
            Output of the regression path.
        attention_weights : torch.Tensor
            Attention weights.
        classification_output : torch.Tensor
            Output of the classification path.
        """

        # ===== Regression path =====
        regression_features = self.convolutional_layers(
            input_tensor
        )  # Apply convolutional layers
        regression_features = regression_features.permute(
            0, 2, 1
        )  # Change shape for LSTM input: [batch, seq_len, features]
        lstm_output, _ = self.lstm_layer(regression_features)  # LSTM output
        attention_output, attention_weights = self.attention_layer(
            lstm_output
        )  # Attention mechanism
        regression_output = self.regression_layers(
            attention_output
        )  # Regression output

        # ===== Classification path =====
        classification_features = self.classification_conv_layers(
            input_tensor
        )  # Convolutional layers
        flattened_features = self.flatten_layer(
            classification_features
        )  # Flatten layer
        classification_output = self.classification_fc_layers(
            flattened_features
        )  # Fully connected layers

        # Combining regression and classification outputs
        combined_output = regression_output * classification_output

        return (
            combined_output,
            regression_output,
            attention_weights,
            classification_output,
        )


class ModelPaperBackward(nn.Module):
    """
    Neural network architecture for backward compatibility.

    Both regression and classification branches enabled

    This model is designed to match a previously model architecture "ModelPaper"
    to ensure compatibility with old models, especially in scenarios
    where reloading a model would otherwise fail due to non-matching architectures.

    Attributes
    ----------
    regression_enabled : bool
        Flag to enable the regression branch of the model.
    classification_enabled : bool
        Flag to enable the classification branch of the model.
    convolutional_layers : nn.Sequential
        Convolutional layers for feature extraction.
    lstm_layer : nn.LSTM
        Bidirectional LSTM layer for sequential data processing.
    regression_layers : nn.Sequential
        Fully connected layers for regression output.
    classification_conv_layers : nn.Sequential
        Convolutional layers for classification feature extraction.
    flatten_layer : nn.Flatten
        Flatten layer to prepare features for the fully connected layers.
    classification_fc_layers : nn.Sequential
        Fully connected layers for classification output.
    """

    def __init__(
        self,
        sequence_length: int,
        num_filters: int,
        kernel_size: int,
        hidden_units: int,
    ) -> None:
        """
        Initializes the ModelPaperBackward architecture.

        Parameters
        ----------
        sequence_length : int
            Length of the output sequence.
        num_filters : int
            Number of filters for convolutional layers.
        kernel_size : int
            Kernel size for convolutional layers.
        hidden_units : int
            Number of hidden units for the LSTM.
        """
        super().__init__()

        self.regression_enabled = True
        self.classification_enabled = True

        # Convolutional layers for regression path
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
        )

        # LSTM layer for sequential data
        self.lstm_layer = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Fully connected layers for regression output with attention mechanism
        self.regression_layers = nn.Sequential(
            AdditiveAttentionBackwards(dim=2 * hidden_units),
            nn.Linear(2 * hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, sequence_length),
        )

        # Convolutional layers for classification path
        self.classification_conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=30, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
        )  # Output shape: [batch, 50, sequence_length - 33]

        # Flatten layer for classification
        self.flatten_layer = nn.Flatten(start_dim=1)

        # Fully connected layers for classification output
        self.classification_fc_layers = nn.Sequential(
            nn.Linear((sequence_length - 33) * 50, 1024),
            nn.ReLU(),
            nn.Linear(1024, sequence_length),
            nn.Sigmoid(),
        )

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).

        Returns
        -------
        combined_output : torch.Tensor
            Combined output of regression and classification paths.
        regression_output : torch.Tensor
            Output of the regression path.
        attention_weights : torch.Tensor
            Attention weights.
        classification_output : torch.Tensor
            Output of the classification path.
        """

        # ===== Regression path =====
        regression_features = self.convolutional_layers(
            input_tensor
        )  # Apply convolutional layers
        regression_features = regression_features.permute(
            0, 2, 1
        )  # Change shape for LSTM input: [batch, seq_len, features]
        lstm_output, _ = self.lstm_layer(regression_features)  # LSTM output
        regression_output = self.regression_layers(
            lstm_output
        )  # Regression output with attention

        # ===== Classification path =====
        classification_features = self.classification_conv_layers(
            input_tensor
        )  # Convolutional layers
        flattened_features = self.flatten_layer(
            classification_features
        )  # Flatten layer
        classification_output = self.classification_fc_layers(
            flattened_features
        )  # Fully connected layers

        # Combining regression and classification outputs
        combined_output = regression_output * classification_output

        # Initialize attention weights as zeros for compatibility
        attention_weights = torch.zeros_like(regression_output)

        return (
            combined_output,
            regression_output,
            attention_weights,
            classification_output,
        )


class ModelOnlyRegression(nn.Module):
    """
    Neural network architecture for appliance disaggregation using only the regression branch.

    This model implements the network architecture described in the paper "Subtask Gated Networks for Non-Intrusive Load Monitoring",
    with the classification branch removed. Only the regression branch is trained
    and used to predict appliance disaggregation.

    Attributes
    ----------
    regression_enabled : bool
        Flag to enable the regression branch of the model.
    classification_enabled : bool
        Flag to indicate that the classification branch is disabled.
    convolutional_layers : nn.Sequential
        Convolutional layers for feature extraction.
    lstm_layer : nn.LSTM
        Bidirectional LSTM layer for sequential data processing.
    attention_layer : AdditiveAttention
        Attention mechanism layer.
    regression_layers : nn.Sequential
        Fully connected layers for regression output.
    """

    def __init__(
        self,
        sequence_length: int,
        num_filters: int,
        kernel_size: int,
        hidden_units: int,
    ) -> None:
        """
        Initializes the ModelOnlyRegression architecture.

        Parameters
        ----------
        sequence_length : int
            Length of the output sequence.
        num_filters : int
            Number of filters for convolutional layers.
        kernel_size : int
            Kernel size for convolutional layers.
        hidden_units : int
            Number of hidden units for the LSTM.
        """
        super().__init__()

        self.regression_enabled = True
        self.classification_enabled = False

        # Convolutional layers for regression path
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
        )

        # LSTM layer for sequential data
        self.lstm_layer = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        # Attention mechanism layer
        self.attention_layer = AdditiveAttention(dim=(2 * hidden_units))

        # Fully connected layers for regression output
        self.regression_layers = nn.Sequential(
            nn.Linear(2 * hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, sequence_length),
        )

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).

        Returns
        -------
        combined_output : torch.Tensor
            Output of the regression path.
        regression_output : torch.Tensor
            Output of the regression path.
        attention_weights : torch.Tensor
            Attention weights.
        classification_output : torch.Tensor
            Dummy classification output for compatibility.
        """
        # ===== Regression path =====
        regression_features = self.convolutional_layers(
            input_tensor
        )  # Apply convolutional layers
        regression_features = regression_features.permute(
            0, 2, 1
        )  # Change shape for LSTM input: [batch, seq_len, features]
        lstm_output, _ = self.lstm_layer(regression_features)  # LSTM output
        context_vector, attention_weights = self.attention_layer(
            lstm_output
        )  # Attention output
        regression_output = self.regression_layers(context_vector)  # Regression output

        # For compatibility, the classification output is set as a dummy
        classification_output = regression_output  # Placeholder to maintain compatibility with other model structures

        combined_output = regression_output
        return (
            combined_output,
            regression_output,
            attention_weights,
            classification_output,
        )


class ModelClassAttention(nn.Module):
    """
    Neural network architecture for appliance disaggregation with attention
    in both regression and classification branches.

    Both the regression and classification branches are enabled and utilize attention mechanisms.

    Attributes
    ----------
    regression_enabled : bool
        Flag to enable the regression branch of the model.
    classification_enabled : bool
        Flag to enable the classification branch of the model.
    convolutional_layers : nn.Sequential
        Convolutional layers for feature extraction in the regression path.
    lstm_layer : nn.LSTM
        Bidirectional LSTM layer for sequential data processing.
    attention_layer : AdditiveAttention
        Attention mechanism layer for both branches.
    regression_layers : nn.Sequential
        Fully connected layers for regression output.
    classification_layers_conv : nn.Sequential
        Convolutional layers for feature extraction in the classification path.
    flatten_layer : nn.Flatten
        Flatten layer to prepare features for the fully connected layers.
    classification_layers_fc : nn.Sequential
        Fully connected layers for classification output.
    """

    def __init__(
        self,
        sequence_length: int,
        num_filters: int,
        kernel_size: int,
        hidden_units: int,
    ) -> None:
        """
        Initializes the ModelClassAttention architecture.

        Parameters
        ----------
        sequence_length : int
            Length of the output sequence.
        num_filters : int
            Number of filters for convolutional layers.
        kernel_size : int
            Kernel size for convolutional layers.
        hidden_units : int
            Number of hidden units for the LSTM.
        """
        super().__init__()

        self.regression_enabled = True
        self.classification_enabled = True

        # Convolutional layers for regression path
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
        )

        # LSTM layer for sequential data processing
        self.lstm_layer = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Attention mechanism layer for both branches
        self.attention_layer = AdditiveAttention(dim=(2 * hidden_units))

        # Fully connected layers for regression output
        self.regression_layers = nn.Sequential(
            nn.Linear(2 * hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, sequence_length),
        )

        # Convolutional layers for feature extraction in classification path
        self.classification_layers_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=30, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
        )  # output [batch, 50, sequence_length - 33]

        # Flatten layer for classification
        self.flatten_layer = nn.Flatten(start_dim=1)  # flatten [batch, (l-33)*50]

        # Fully connected layers for classification output
        self.classification_layers_fc = nn.Sequential(
            nn.Linear((sequence_length - 33) * 50 + 2 * hidden_units, 1024),
            nn.ReLU(),
            nn.Linear(1024, sequence_length),
            nn.Sigmoid(),
        )

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).

        Returns
        -------
        combined_output : torch.Tensor
            Combined output of regression and classification paths.
        regression_output : torch.Tensor
            Output of the regression path.
        attention_weights : torch.Tensor
            Attention weights.
        classification_output : torch.Tensor
            Output of the classification path.
        """
        # ===== Regression path =====
        regression_features = self.convolutional_layers(input_tensor)
        regression_features = regression_features.permute(
            0, 2, 1
        )  # Adjust shape for LSTM input: [batch, seq_len, features]
        lstm_output, _ = self.lstm_layer(regression_features)
        context_vector, attention_weights = self.attention_layer(lstm_output)
        regression_output = self.regression_layers(context_vector)

        # ===== Classification path =====
        classification_features_conv = self.classification_layers_conv(input_tensor)
        classification_features_flat = self.flatten_layer(
            classification_features_conv
        )  # classification_features_conv.view(classification_features_conv.size(0), -1)
        classification_features_combined = torch.cat(
            (classification_features_flat, context_vector), dim=1
        )
        classification_output = self.classification_layers_fc(
            classification_features_combined
        )

        # Combine regression and classification outputs
        combined_output = regression_output * classification_output

        return (
            combined_output,
            regression_output,
            attention_weights,
            classification_output,
        )
