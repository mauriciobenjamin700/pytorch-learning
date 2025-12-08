from .get_device import get_device
from .neural_network import NeuralNetwork


def build_model():
    """
    Build and return the neural network model.

    Returns:
        model (NeuralNetwork): The neural network model.
    """
    device = get_device()
    model = NeuralNetwork().to(device)
    return model
