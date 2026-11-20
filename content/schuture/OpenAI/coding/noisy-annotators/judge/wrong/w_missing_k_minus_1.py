# expect: WA
# Uses the binary log-odds log(r / (1 - r)), dropping the K - 1 factor, so for K = 3 annotators at r < 0.5 get weight 0 and their samples are dropped.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

N_TRAIN, N_TEST, N_FEATURES, N_CLASSES, N_ANNOTATORS = 2000, 1000, 60, 3, 6


def make_dataset(seed):
    """Returns (X_train, y_train_true, X_test, y_test, annotations), built deterministically from seed.
    y_train_true, shape (2000,), is the ground-truth class of every training point: use it only to
    check your answer to Part 1, never to build a training label. annotations, shape (2000, 6), has
    entries in {-1, 0, 1, 2}."""
    rng = np.random.default_rng(seed)
    X, y = make_classification(n_samples=N_TRAIN + N_TEST, n_features=N_FEATURES, n_informative=40,
                                n_redundant=10, n_classes=N_CLASSES, n_clusters_per_class=2,
                                class_sep=1.6, flip_y=0.01, random_state=seed)
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]

    annotations = np.full((N_TRAIN, N_ANNOTATORS), -1)
    for i in range(N_TRAIN):
        k = rng.integers(2, 5)                                    # 2 to 4 annotators label this sample
        for j in rng.choice(N_ANNOTATORS, size=int(k), replace=False):
            if j == 4:                                             # NOTE: ignores X_train and y_train entirely
                annotations[i, j] = rng.integers(0, N_CLASSES)
            elif j == 5:                                           # NOTE: systematically biased, not noisy
                annotations[i, j] = 0
            elif rng.random() < 0.92:
                annotations[i, j] = y_train[i]
            else:
                others = [c for c in range(N_CLASSES) if c != y_train[i]]
                annotations[i, j] = rng.choice(others)
    return X_train, y_train, X_test, y_test, annotations


def aggregate_labels(annotations, n_classes, weights=None):
    n, A = annotations.shape
    if weights is None:
        weights = np.ones(A)
    scores = np.zeros((n, n_classes))
    for j in range(A):
        labeled = annotations[:, j] != -1
        scores[labeled, annotations[labeled, j]] += weights[j]
    has_label = scores.sum(axis=1) > 0
    labels = np.argmax(scores, axis=1)               # NOTE: argmax keeps the FIRST max -> ties go to class 0
    return labels, has_label


def annotator_reliability(annotations, n_classes):
    n, A = annotations.shape
    r = np.zeros(A)
    for j in range(A):
        peers = np.delete(annotations, j, axis=1)
        majority, has_majority = aggregate_labels(peers, n_classes)
        labeled_by_j = annotations[:, j] != -1
        mask = labeled_by_j & has_majority
        r[j] = np.mean(annotations[mask, j] == majority[mask]) if mask.any() else 0.0
    return r


def flag_bad_annotators(r):
    return r < r.mean() - r.std()


def fit_and_score(X_train, labels, has_label, X_test, y_test):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train[has_label], labels[has_label])
    return accuracy_score(y_test, clf.predict(X_test))


def reliability_weights(r, n_classes):
    r = np.clip(r, 1e-2, 1 - 1e-2)                    # NOTE: avoids log(0) at r == 0 or r == 1
    w = np.log(r / (1 - r))
    return np.clip(w, 0.0, None)                      # NOTE: chance-level or worse -> weight 0, not a negative vote
