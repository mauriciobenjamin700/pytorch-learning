import torch
from torch import nn

from .neural_network import NeuralNetwork


def get_model_optimizer_params(
    model: NeuralNetwork
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """
    Given a neural network model, return the loss function and optimizer.

    Args:
        model (NeuralNetwork): The neural network model.

    Returns:
        tuple: A tuple containing the loss
            function and optimizer
    """

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return loss_fn, optimizer
