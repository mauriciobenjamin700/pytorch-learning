from typing import Any
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_data() -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """
    Load FashionMNIST dataset and return training and test data loaders.

    Args:
        batch_size (int): The number of samples per batch to load.

    Returns:
        tuple: A tuple containing the
            training and test data loaders.
    """
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, test_data


def get_dataloader(
    batch_size: int = 64
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Create and return training and test data loaders for the FashionMNIST
        dataset.

    Args:
        batch_size (int): The number of samples per batch to load.

    Returns:
        tuple: A tuple containing the
    """
    training_data, test_data = load_data()

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


__all__ = ["load_data"]
