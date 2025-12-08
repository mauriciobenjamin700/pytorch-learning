from torch import nn
import torch


def save_model(model: nn.Module, path: str = "model.pth") -> None:
    """
    Save the model's state dictionary to the specified path.

    Args:
        model (nn.Module): The neural network model to save.
        path (str): The file path to save the model's state dictionary.

    Returns:
        None
    """
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str = "model.pth") -> nn.Module:
    """
    Load the model's state dictionary from the specified path.

    Args:
        model (nn.Module): The neural network model to load the state
            dictionary into.
        path (str): The file path to load the model's state dictionary from.

    Returns:
        model (nn.Module): The neural network model with loaded state
            dictionary.
    """
    model.load_state_dict(torch.load(path, weights_only=True))
    return model
