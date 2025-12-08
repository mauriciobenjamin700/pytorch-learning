import torch
from torch import nn


class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network for image classification.

    Args:
        None

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the network.

    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
