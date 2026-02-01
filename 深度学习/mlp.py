import torch
from torch import nn
from d2l import torch as d2l
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:", device)

    batch_size = 256
    num_epochs = 100
    lr = 0.1
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), 
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    net.to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            X,y = X.to(device), y.to(device)
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            print(f'Epoch {epoch + 1}, Loss: {l.item()}')
    
    

if __name__ == "__main__":
    # Windows 多进程 DataLoader 需要这个保护
    main()