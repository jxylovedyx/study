import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class VGG11(nn.Module):
    """VGG-11: 8 conv layers + 3 FC layers."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


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
    num_epochs: int = 10,
    lr: float = 0.01,
) -> None:
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

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

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (y_hat.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        test_acc = evaluate_accuracy(net, test_iter, device)
        print(
            f"epoch {epoch + 1:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"test acc {test_acc:.4f}"
        )


if __name__ == "__main__":
    batch_size = 32
    resize = 224
    num_epochs = 5
    lr = 0.01

    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
    ])

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
    net = VGG11(num_classes=10, in_channels=1)

    # Quick shape check
    X = torch.randn(2, 1, 224, 224)
    print("Output shape:", net(X).shape)

    train(net, train_iter, test_iter, device=device, num_epochs=num_epochs, lr=lr)
