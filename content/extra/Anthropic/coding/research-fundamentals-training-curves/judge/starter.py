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
    """Mean squared error of model on (X, y), computed in eval mode without building a graph.
    Must leave model in the same mode (train / eval) it was in before the call."""
    raise NotImplementedError


def train(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    """Minibatch SGD on the mean squared error; return {"train": [...], "val": [...], "train_eval": [...]}."""
    raise NotImplementedError


# ---------------------------------------------------------------- Part 2: the colleague's code, 3 bugs
def train_colleague(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mse = lambda pred, target: ((pred - target) ** 2).mean()
    n = X_train.shape[0]
    history = {"train": [], "val": []}
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = mse(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train"].append(total / n)
        with torch.no_grad():
            history["val"].append(mse(model(X_val), y_val).item())
    return history


# ---------------------------------------------------------------- Part 3
def linear_loss(theta, X, y):                  # given
    return ((X @ theta[:-1] + theta[-1] - y) ** 2).mean()


def linear_grad(theta, X, y):
    """Gradient of linear_loss with respect to theta, derived by hand (no autograd). Shape (9,)."""
    raise NotImplementedError


def fd_grad(f, theta, eps):
    """Central differences: entry i is (f(theta + eps * e_i) - f(theta - eps * e_i)) / (2 * eps)."""
    raise NotImplementedError


def gd_linear(X, y, lr, steps):
    """Full-batch GD with torch.optim.SGD on nn.Linear(8, 1) in float64 from zero; return (steps + 1, 9) thetas."""
    raise NotImplementedError
