import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def nin_block(in_channels, out_channels, kernel_size, stride, padding, apply_last_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1)
    ]
    if apply_last_relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def nin():
    return nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, stride=1, padding=1, apply_last_relu=False),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
         xscale="linear", yscale="linear", fmts=("-", "m--"),
         figsize=(6, 4), save_path="nin_train_curve.png"):
    if legend is None:
        legend = []
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    if not isinstance(X[0], (list, tuple)):
        X = [X] * len(Y)
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()

def train(num_epochs=10, batch_size=128, lr=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_set = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = nin().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    train_losses = []
    train_accs = []
    test_accs = []
    epochs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            if epoch == 0 and total == 0:  # 第一个epoch第一个batch
                print("x mean/std:", x.mean().item(), x.std().item())
                print("logits mean/std:", logits.mean().item(), logits.std().item())
                print("logits[0]:", logits[0].detach().cpu())

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total if total > 0 else 0.0
        test_acc = evaluate(model, test_loader, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epochs.append(epoch + 1)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "nin_fashionmnist.pth")
    print("Model saved to nin_fashionmnist.pth")

    plot(
        X=[epochs, epochs],
        Y=[train_losses, test_accs],
        xlabel="epoch",
        ylabel="value",
        legend=["train loss", "test acc"],
        xlim=[1, num_epochs],
        fmts=("-", "m--"),
        save_path="nin_train_curve.png"
    )
    print("Train curve saved to nin_train_curve.png")

    plot(
        X=[epochs, epochs],
        Y=[train_accs, test_accs],
        xlabel="epoch",
        ylabel="accuracy",
        legend=["train acc", "test acc"],
        xlim=[1, num_epochs],
        ylim=[0, 1],
        fmts=("-", "m--"),
        save_path="nin_acc_curve.png"
    )
    print("Acc curve saved to nin_acc_curve.png")

if __name__ == "__main__":
    train()
