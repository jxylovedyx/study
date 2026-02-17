import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from datetime import datetime
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vgg_block(num_convs: int, in_channels: int, out_channels: int) -> nn.Sequential:
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch, in_channels: int = 1, num_classes: int = 10) -> nn.Sequential:
    conv_blks = []
    for num_convs, out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), 
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, num_classes),
    )


def evaluate_accuracy(net: nn.Module, data_iter: DataLoader, device: torch.device) -> float:
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            pred = net(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def train(
    net: nn.Module,
    train_iter: DataLoader,
    test_iter: DataLoader,
    device: torch.device,
    num_epochs: int = 30,
    lr: float = 0.01,
):
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    print(f"Training on: {device}")
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (y_hat.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        test_acc = evaluate_accuracy(net, test_iter, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"epoch {epoch + 1:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"test acc {test_acc:.4f}"
        )

    return history


def plot_history(history: dict, save_path: Optional[str] = None) -> str:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="train acc")
    plt.plot(epochs, history["test_acc"], marker="s", label="test acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy Curve")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    project_root = Path(__file__).resolve().parent.parent
    if save_path is None:
        save_dir = project_root / "CNN"
        save_name = f"vgg_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = save_dir / save_name
    else:
        output_path = Path(save_path)
        if not output_path.is_absolute():
            output_path = project_root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved curve to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    lr = 0.01
    num_epochs = 10
    batch_size = 64
    resize = 224

    transform = transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True,
    )

    train_iter = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_iter = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = vgg(conv_arch=conv_arch, in_channels=1, num_classes=10)
    net = net.to(device)
    print("Using device:", device)

    X = torch.randn(2, 1, resize, resize, device=device)
    print("Output shape:", net(X).shape)

    history = train(
        net=net,
        train_iter=train_iter,
        test_iter=test_iter,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
    )
    plot_history(history)
