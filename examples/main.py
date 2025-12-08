import torch
from .build_model import build_model
from .get_device import get_device
from .load_data import load_data, get_dataloader
from .neural_network import NeuralNetwork
from .optimizing_model_params import get_model_optimizer_params
from .storage import save_model
from .train_model import train_model


def main() -> None:

    _, test_data = load_data()
    train_dataloader, test_dataloader = get_dataloader()

    model: NeuralNetwork = build_model()

    loss_fn, optimizer = get_model_optimizer_params(model)

    device = get_device()

    model = train_model(
        100,
        train_dataloader,
        test_dataloader,
        model,
        loss_fn,
        optimizer,
        device
    )

    save_model(model, "model.pth")

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == "__main__":
    main()
