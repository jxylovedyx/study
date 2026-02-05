import torch
from torch import nn
from d2l import torch as d2l

num_inputs=784
num_hiddens1=256
num_hiddens2=256
num_outputs=10
dropout1=0.2
dropout2=0.5

batch_size = 256
num_epochs = 10


def net():
    return nn.Sequential(nn.Linear(num_inputs, num_hiddens1),
                        nn.ReLU(),
                        nn.Dropout(dropout1),
                        nn.Linear(num_hiddens1, num_hiddens2),
                        nn.ReLU(),
                        nn.Dropout(dropout2),
                        nn.Linear(num_hiddens2, num_outputs))


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:", device)
    model = net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X, y = torch.randn(batch_size, num_inputs), 
            torch.randint(0, num_outputs, (batch_size,))
            X, y = X.to(device), y.to(device)
        y_hat = model(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        
    