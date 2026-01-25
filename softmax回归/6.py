import torch
from IPython import display
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size = 256
lr = 0.1
num_epochs = 20
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:", device)


W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), device=device, requires_grad=True)
b = torch.zeros(num_outputs, device=device, requires_grad=True)

def softmax(X):
    X = X - X.max(dim=1, keepdim=True).values  # 防exp溢出
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, num_inputs)), W) + b)

def cross_entropy(y_hat, y):
    eps = 1e-12
    p = y_hat[range(len(y_hat)), y].clamp(min=eps)  # 防log(0)
    return -torch.log(p)

def accuracy(y_hat, y):
    # y_hat: (batch, num_classes) 或 (batch,)
    # y:     (batch,)
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)      # (batch,)
    y_hat = y_hat.type_as(y)             # dtype 对齐
    return (y_hat == y).float().mean().item()

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y) * y.shape[0], y.shape[0])
    return metric[0] / metric[1]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols,    
                                              figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes(xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts    
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            self.axes[0].plot(x, y, self.fmts[i])
        display.display(self.fig)
        display.clear_output(wait=True)

    def config_axes(self, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        ax = self.axes[0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if legend: ax.legend(legend)
        ax.grid(True)



def train_epoch_ch3(net, train_iter, loss, updater):  
    if isinstance(net, torch.nn.Module):
        net.train()  
    metric = Accumulator(3)  
    for X, y in train_iter:
        X,y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y).sum()

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
        else:
            l.backward()
            updater(X.shape[0])

        metric.add(float(l), accuracy(y_hat, y)*y.numel(), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
    
def train_ch3(net, train_iter, test_iter, loss, updater):  
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs): 
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(f"epoch {epoch+1}: loss {train_metrics[0]:.4f}, train acc {train_metrics[1]:.4f}, test acc {test_acc:.4f}")

    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc

def predict_ch3(net, test_iter, n=6):  
    for X, y in test_iter:
        break
    X, y = X.to(device), y.to(device)
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_fashion_mnist(X[0:n], titles[0:n])

if __name__ == "__main__":
    updater = torch.optim.SGD([W, b], lr=lr)
    train_ch3(net, train_iter, test_iter, cross_entropy, updater)
    predict_ch3(net, test_iter)
    plt.savefig("train_curve.png", dpi=200, bbox_inches="tight")
    plt.show()

