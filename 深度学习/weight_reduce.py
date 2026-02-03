def init_params():
    w = torch.normal(0, 1, size=(max_degree, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2 

def train(lambd):
    w,b = init_params()
    net = lambda x: d2l.linreg(x, w, b)
    loss = d2l.squared_loss
    num_epochs, lr = 100, 0.01
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], legend=['train'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l = l.mean()
            l.backward()
            d2l.sgd([w, b], lr, 1)
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(train_features), train_labels)
        animator.add(epoch + 1, (float(train_l.mean()),))