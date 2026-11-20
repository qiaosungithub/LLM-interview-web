# expect: WA
# Part 1: evaluate always ends with model.train(), even when the model came in eval mode.
import torch
import torch.nn as nn


def make_data(n, seed):
    """x ~ N(0, I_8) in float32; y = relu(x0 - x1) - relu(x2) + 0.5 * x3 + 0.1 * noise."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 8, generator=g)
    clean = torch.relu(X[:, 0] - X[:, 1]) - torch.relu(X[:, 2]) + 0.5 * X[:, 3]
    y = clean + 0.1 * torch.randn(n, generator=g)
    return X, y


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))


X_train, y_train = make_data(1000, seed=0)
X_val, y_val = make_data(1000, seed=1000)


# ---------------------------------------------------------------- Part 1
def evaluate(model, X, y):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = ((model(X).squeeze(-1) - y) ** 2).mean().item()
    model.train()
    return loss


def train(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = X_train.shape[0]
    history = {"train": [], "val": [evaluate(model, X_val, y_val)], "train_eval": [evaluate(model, X_train, y_train)]}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            opt.zero_grad()
            loss = ((model(xb).squeeze(-1) - yb) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train"].append(total / n)
        history["val"].append(evaluate(model, X_val, y_val))
        history["train_eval"].append(evaluate(model, X_train, y_train))
    return history


# ---------------------------------------------------------------- Part 2 (the page's train_colleague_fixed)
def train_colleague(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mse = lambda pred, target: ((pred.squeeze(-1) - target) ** 2).mean()   # B: (n, 1) - (n,) broadcasts to (n, n)
    n = X_train.shape[0]
    history = {"train": [], "val": []}
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()                                               # A: backward() adds to .grad
            loss = mse(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train"].append(total / n)
        model.eval()                                                      # C: no_grad() does not turn dropout off
        with torch.no_grad():
            history["val"].append(mse(model(X_val), y_val).item())
        model.train()
    return history


# ---------------------------------------------------------------- Part 3
def linear_loss(theta, X, y):                  # given
    return ((X @ theta[:-1] + theta[-1] - y) ** 2).mean()


def linear_grad(theta, X, y):
    n = X.shape[0]
    r = X @ theta[:-1] + theta[-1] - y
    return torch.cat([2.0 / n * (X.T @ r), (2.0 / n * r.sum()).reshape(1)])


def fd_grad(f, theta, eps):
    g = torch.zeros_like(theta)
    for i in range(theta.numel()):
        e = torch.zeros_like(theta)
        e[i] = eps
        g[i] = (f(theta + e) - f(theta - e)) / (2 * eps)
    return g


def gd_linear(X, y, lr, steps):
    model = nn.Linear(X.shape[1], 1).to(X.dtype)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    thetas = [torch.cat([model.weight.detach().reshape(-1), model.bias.detach()]).clone()]
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(X).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        opt.step()
        thetas.append(torch.cat([model.weight.detach().reshape(-1), model.bias.detach()]).clone())
    return torch.stack(thetas)
