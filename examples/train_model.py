from typing import Any, Literal
import torch
from torch import nn
from torch.utils.data import DataLoader


def train(
    dataloader: DataLoader[Any],
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Literal["cpu", "cuda"] = "cpu"
) -> nn.Module:
    """
    Train the model for one epoch.

    Args:
        dataloader (DataLoader[Any]): The data loader for training data.
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): The device to run the training on.

    Returns:
        model (nn.Module): The trained model.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return model


def test(
    dataloader: DataLoader[Any],
    model: nn.Module,
    loss_fn: nn.Module,
    device: Literal["cpu", "cuda"] = "cpu"
) -> tuple[float, float]:
    """
    Test the model and print the accuracy and average loss.

    Args:
        dataloader (DataLoader[Any]): The data loader for test data.
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.
        device (str): The device to run the testing on.

    Returns:
        tuple: The accuracy and average loss.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%")
    print(f" Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def train_model(
    epochs: int,
    train_dataloader: DataLoader[Any],
    test_dataloader: DataLoader[Any],
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Literal["cpu", "cuda"] = "cpu"
) -> nn.Module:
    """
    Train and test the model for a given number of epochs.

    Args:
        epochs (int): The number of epochs to train.
        train_dataloader (DataLoader[Any]): The data loader for training data.
        test_dataloader (DataLoader[Any]): The data loader for test data.
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): The device to run the training and testing on.

    Returns:
        model (nn.Module): The trained model.
    """
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model = train(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            device
        )
        test(
            test_dataloader,
            model,
            loss_fn,
            device
        )
    print("Done!")
    return model
