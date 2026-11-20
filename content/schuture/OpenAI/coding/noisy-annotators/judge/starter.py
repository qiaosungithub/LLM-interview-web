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


def aggregate_labels(annotations: np.ndarray, n_classes: int,
                      weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Part 1 (a): weighted vote per sample, ties to the smaller class; returns (labels, has_label)."""
    raise NotImplementedError


def annotator_reliability(annotations: np.ndarray, n_classes: int) -> np.ndarray:
    """Part 1 (b): fraction of each annotator's samples on which it agrees with the vote of the others."""
    raise NotImplementedError


def flag_bad_annotators(r: np.ndarray) -> np.ndarray:
    """Part 1 (c): True where a score is more than one standard deviation below the mean."""
    raise NotImplementedError


def fit_and_score(X_train: np.ndarray, labels: np.ndarray, has_label: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Part 2: fits LogisticRegression(max_iter=2000) on the labelled rows; returns test accuracy."""
    raise NotImplementedError


def reliability_weights(r: np.ndarray, n_classes: int) -> np.ndarray:
    """Part 3: r: (A,), values in [0, 1]. Returns weights of shape (A,), each >= 0."""
    raise NotImplementedError
