import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # C1: 1x28x28 -> 6x28x28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # S2: 6x28x28 -> 6x14x14
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: 6x14x14 -> 16x10x10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            # S4: 16x10x10 -> 16x5x5
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),          # 16x5x5 -> 400
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate_accuracy(net: nn.Module, 
                      data_iter: DataLoader, 
                      device: torch.device) -> float:
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


def train(net: nn.Module, 
          train_iter: DataLoader, 
          test_iter: DataLoader, 
          device: torch.device,
          num_epochs: int = 10, lr: float = 0.9) -> None:
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on: {device}")
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.numel()

        train_loss = total_loss / total
        train_acc = correct / total
        test_acc = evaluate_accuracy(net, test_iter, device)
        print(
            f"epoch {epoch + 1:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"test acc {test_acc:.4f}"
        )


if __name__ == "__main__":
    batch_size = 256
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LeNet(num_classes=10)
    train(net, train_iter, test_iter, device=device, num_epochs=100, lr=0.9)
