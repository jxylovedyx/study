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

def evaluate_accuracy_gpu(net: nn.Module, 
                          data_iter: DataLoader, 
                          device: torch.device) -> float:
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（词ID、有效长度、类型ID）三元组
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add((net(X).argmax(dim=1) == y).sum().item(), y.numel())
    return metric[0] / metric[1] 


    

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

def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch + 1}, loss {train_loss_sum / n:.4f}, '
              f'train acc {train_acc_sum / n:.3f}, test acc {test_acc:.3f}')
