"""Helpers for tests.py: the statement's data generator and plain-loop references for every function."""
import math

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

N_TRAIN, N_TEST, N_FEATURES, N_CLASSES, N_ANNOTATORS = 2000, 1000, 60, 3, 6


def make_dataset(seed):
    """Copied from the statement."""
    rng = np.random.default_rng(seed)
    X, y = make_classification(n_samples=N_TRAIN + N_TEST, n_features=N_FEATURES, n_informative=40,
                               n_redundant=10, n_classes=N_CLASSES, n_clusters_per_class=2,
                               class_sep=1.6, flip_y=0.01, random_state=seed)
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]

    annotations = np.full((N_TRAIN, N_ANNOTATORS), -1)
    for i in range(N_TRAIN):
        k = rng.integers(2, 5)
        for j in rng.choice(N_ANNOTATORS, size=int(k), replace=False):
            if j == 4:
                annotations[i, j] = rng.integers(0, N_CLASSES)
            elif j == 5:
                annotations[i, j] = 0
            elif rng.random() < 0.92:
                annotations[i, j] = y_train[i]
            else:
                others = [c for c in range(N_CLASSES) if c != y_train[i]]
                annotations[i, j] = rng.choice(others)
    return X_train, y_train, X_test, y_test, annotations


def aggregate(annotations, n_classes, weights=None):
    """Plain loops. Returns (labels, has_label) as lists; labels[i] is None where has_label[i] is False. Ties go to
    the smaller class. Exact when the weights are multiples of a power of two, as in the tests."""
    labels, has = [], []
    for row in np.asarray(annotations).tolist():
        tot = [0.0] * n_classes
        for j, a in enumerate(row):
            if a != -1:
                tot[a] += 1.0 if weights is None else float(weights[j])
        ok = sum(tot) > 0
        has.append(ok)
        labels.append(tot.index(max(tot)) if ok else None)
    return labels, has


def reliability(annotations, n_classes):
    """Plain loops. Returns a list of floats, or None for an annotator with no sample that counts."""
    ann = np.asarray(annotations).tolist()
    A = len(ann[0])
    out = []
    for j in range(A):
        agree = total = 0
        for row in ann:
            if row[j] == -1:
                continue
            peers = [a for k, a in enumerate(row) if k != j and a != -1]
            if not peers:
                continue
            counts = [peers.count(c) for c in range(n_classes)]
            total += 1
            agree += row[j] == counts.index(max(counts))
        out.append(agree / total if total else None)
    return out


def flag_readings(r):
    """(flags with the population standard deviation, flags with the sample one, smallest distance of any r_j from
    either threshold). The statement does not say which standard deviation, so tests use only r where both agree."""
    r = [float(v) for v in r]
    A = len(r)
    mean = sum(r) / A
    ss = sum((v - mean) ** 2 for v in r)
    t_pop, t_smp = mean - math.sqrt(ss / A), mean - math.sqrt(ss / (A - 1))
    gap = min(min(abs(v - t_pop), abs(v - t_smp)) for v in r)
    return [v < t_pop for v in r], [v < t_smp for v in r], gap


def log_odds_weights(r, n_classes):
    """The page's weights, used only for messages."""
    r = np.clip(np.asarray(r, dtype=float), 1e-2, 1 - 1e-2)
    return np.clip(np.log(r * (n_classes - 1) / (1 - r)), 0.0, None)


def fit_and_score(X_train, labels, has_label, X_test, y_test):
    has_label = np.asarray(has_label, dtype=bool)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train[has_label], np.asarray(labels)[has_label])
    return float(np.mean(clf.predict(X_test) == y_test))
