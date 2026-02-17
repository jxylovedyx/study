import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),  # 卷积层1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                   # 池化层1
            nn.Conv2d(96, 256, kernel_size=5, padding=2),            # 卷积层2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                   # 池化层2
            nn.Conv2d(256, 384, kernel_size=3, padding=1),           # 卷积层3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),           # 卷积层4
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),           # 卷积层5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                   # 池化层3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                             # 展平层
            nn.LazyLinear(4096),                                      # 全连接层1（自动推断输入维度）
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),                                    # 全连接层2
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),                             # 输出层
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
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
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
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
    batch_size = 128
    resize = 224
    num_epochs = 100
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
    net = AlexNet(num_classes=10)
    train(net, train_iter, test_iter, device=device, num_epochs=num_epochs, lr=lr)
