import torch


def get_device() -> str:
    """
    Get the current device type (e.g., 'cuda', 'mps', 'cpu').

    Returns:
        str: The type of the current device.
    """
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available() else "cpu"
    )
    print(f"Using {device} device")
    return device


__all__ = ["get_device"]
