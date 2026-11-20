"""Helpers for tests.py: the statement's data and model, the page's Part 1 training loop, and Part 3 references."""
import torch
import torch.nn as nn

F64 = torch.float64


def make_data(n, seed):
    """The statement's generator, unchanged."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 8, generator=g)
    clean = torch.relu(X[:, 0] - X[:, 1]) - torch.relu(X[:, 2]) + 0.5 * X[:, 3]
    y = clean + 0.1 * torch.randn(n, generator=g)
    return X, y


def make_model(seed):
    """The statement's model, unchanged."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))


def evaluate(model, X, y):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = ((model(X).squeeze(-1) - y) ** 2).mean().item()
    model.train(was_training)
    return loss


def train(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    """The page's Part 1 loop: the order of every random draw is fixed by the statement."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = X_train.shape[0]
    history = {"train": [], "val": [evaluate(model, X_val, y_val)], "train_eval": [evaluate(model, X_train, y_train)]}
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = ((model(X_train[idx]).squeeze(-1) - y_train[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train"].append(total / n)
        history["val"].append(evaluate(model, X_val, y_val))
        history["train_eval"].append(evaluate(model, X_train, y_train))
    return history


def linear_loss(theta, X, y):
    """The statement's given loss."""
    return ((X @ theta[:-1] + theta[-1] - y) ** 2).mean()


def autograd_grad(theta, X, y):
    """The gradient of linear_loss by autograd: independent of any hand derivation."""
    t = theta.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(linear_loss(t, X, y), t)
    return g


def central_differences(f, theta, eps):
    """The statement's definition, entry by entry."""
    out = []
    for i in range(theta.numel()):
        e = torch.zeros_like(theta)
        e[i] = eps
        out.append(float(f(theta + e) - f(theta - e)) / (2 * eps))
    return torch.tensor(out, dtype=F64)


def gd_recurrence(X, y, lr, steps):
    """theta_{t+1} = theta_t - lr * grad L(theta_t) from theta_0 = 0, grad = (2/n) [X, 1]^T r, written out by hand."""
    n = X.shape[0]
    Xt = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)
    theta = torch.zeros(Xt.shape[1], dtype=X.dtype)
    out = [theta]
    for _ in range(steps):
        theta = theta - lr * (2.0 / n) * (Xt.T @ (Xt @ theta - y))
        out.append(theta)
    return torch.stack(out)
