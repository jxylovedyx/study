import torch
from torch import nn  
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size = 256
num_epochs = 30
lr = 0.1
num_inputs = 784
num_outputs = 10  
num_hiddens = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:", device)

W1 =  nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True, device=device) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True, device=device))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True, device=device) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True, device=device))

params = [W1, b1, W2, b2]

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)

def net(X):
    X = X.to(device)
    X = X.reshape((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

loss = nn.CrossEntropyLoss()
def updater(batch_size):
    d2l.sgd(params, lr, batch_size)
    
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(net, test_iter)
d2l.plt.show()