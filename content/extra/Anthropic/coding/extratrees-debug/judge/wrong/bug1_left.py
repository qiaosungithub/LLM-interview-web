# expect: WA
# Bug 1 left in: a node with exactly min_samples_split samples becomes a leaf (<= instead of <).
"""Extremely Randomized Trees (Geurts, Ernst & Wehenkel, 2006) for classification.

The docstrings in this file are the specification.

Growing a tree.  Every tree is grown on the whole training set (no bootstrap).
A node that holds the sample subset S becomes a leaf if any of these is true:

  (a) |S| < min_samples_split,
  (b) all labels in S are equal,
  (c) every feature is constant in S.

Otherwise the node is split:

  1. Let A be the set of features that are not constant in S.  Draw
     k = min(max_features, |A|) distinct features from A uniformly at random.
  2. For each drawn feature f, draw one cut-point uniformly from
     [min_S f, max_S f).  A draw equal to min_S f is replaced by max_S f.
  3. Keep the candidate (f, cut) with the highest score (see split_scores);
     on ties, the one drawn first.
  4. Samples with x[f] < cut go to the left child, all other samples go to the
     right child.  By construction both children receive at least one sample.

Storage.  Nodes are numbered in pre-order (the root is 0, and the left subtree
of a node is numbered before its right subtree) and kept in flat arrays of
length n_nodes:

  feature_[i]    split feature of node i, -1 for a leaf
  threshold_[i]  cut-point of node i, nan for a leaf
  left_[i]       id of the left child, -1 for a leaf
  right_[i]      id of the right child, -1 for a leaf
  value_[i]      shape (n_classes,): how many training samples of each class
                 reached node i
"""
import numpy as np


def entropy(counts):
    """Shannon entropy, in bits, of each row of `counts`.

    counts: array of shape (..., m) holding non-negative counts.
    Returns an array of shape (...).  0 * log(0) counts as 0, and a row that
    is all zeros has entropy 0.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=-1)


def split_scores(go_left, onehot):
    """Score k candidate splits of the same n samples.

    go_left: bool array (n, k); go_left[i, j] is True if sample i goes to the
             left child under candidate j.
    onehot:  array (n, n_classes), the one-hot labels of the samples.

    Returns an array of shape (k,) with

        score = 2 * I / (H_split + H_class)

    where H_class is the entropy of the labels, H_split is the entropy of the
    left/right partition and I is their mutual information, i.e. H_class minus
    the size-weighted mean of the label entropies of the two children.
    The score lies in [0, 1].  It is 0 when the denominator is 0.
    """
    n = len(onehot)
    total = onehot.sum(axis=0)
    left = go_left.T.astype(float) @ onehot
    right = total - left
    n_left = left.sum(axis=1)
    n_right = n - n_left
    h_class = entropy(total)
    h_split = entropy(np.stack([n_left, n_right], axis=1))
    gain = h_class - (n_left * entropy(left) + n_right * entropy(right)) / n
    denom = h_split + h_class
    return np.divide(2.0 * gain, denom, out=np.zeros_like(gain), where=denom > 0)


class ExtraTree:
    """One extremely randomized classification tree.

    max_features:      number of candidate features drawn at each node (K >= 1).
    min_samples_split: a node with fewer samples is never split (n_min >= 2).
    seed:              anything np.random.default_rng accepts.
    """

    def __init__(self, max_features, min_samples_split=2, seed=None):
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        """Grow the tree on X (n, d) and labels y (n,), n >= 1.  Returns self.

        Sets classes_ (the sorted distinct labels) and the node arrays that
        the module docstring describes.
        """
        X = np.asarray(X, dtype=float)
        self.classes_, codes = np.unique(y, return_inverse=True)
        onehot = np.eye(len(self.classes_))[codes]
        self._rng = np.random.default_rng(self.seed)
        self._nodes = []
        self._grow(X, onehot, np.arange(len(X)))
        feature, threshold, left, right, value = zip(*self._nodes)
        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.left_ = np.array(left)
        self.right_ = np.array(right)
        self.value_ = np.array(value)
        return self

    def _grow(self, X, onehot, idx):
        """Append the subtree for the samples `idx`; return the id of its root."""
        node = len(self._nodes)
        counts = onehot[idx].sum(axis=0)
        self._nodes.append([-1, np.nan, -1, -1, counts])
        split = self._pick_split(X, onehot, idx, counts)
        if split is not None:
            feature, cut = split
            go_left = X[idx, feature] < cut
            left = self._grow(X, onehot, idx[go_left])
            right = self._grow(X, onehot, idx[~go_left])
            self._nodes[node][:4] = [feature, cut, left, right]
        return node

    def _pick_split(self, X, onehot, idx, counts):
        """Return (feature, cut) for the node holding `idx`, or None for a leaf."""
        if len(idx) <= self.min_samples_split or counts.max() == counts.sum():
            return None
        lo, hi = X[idx].min(axis=0), X[idx].max(axis=0)
        if np.all(lo == hi):
            return None
        candidates = np.flatnonzero(lo < hi)
        k = min(self.max_features, len(candidates))
        feats = self._rng.choice(candidates, size=k, replace=False)
        cuts = self._rng.uniform(lo[feats], hi[feats])
        cuts = np.where(cuts > lo[feats], cuts, hi[feats])
        go_left = X[np.ix_(idx, feats)] < cuts
        best = np.argmax(split_scores(go_left, onehot[idx]))
        return int(feats[best]), float(cuts[best])

    def apply(self, X):
        """Id of the leaf that each row of X ends up in, shape (n,).

        Every sample starts at the root.  At an internal node i it moves to
        left_[i] if x[feature_[i]] < threshold_[i], otherwise to right_[i].
        """
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        while True:
            rows = np.flatnonzero(self.feature_[node] >= 0)
            if rows.size == 0:
                return node
            at = node[rows]
            go_left = X[rows, self.feature_[at]] < self.threshold_[at]
            node[rows] = np.where(go_left, self.left_[at], self.right_[at])

    def predict_proba(self, X):
        """Class frequencies in the leaf of each sample, shape (n, n_classes)."""
        counts = self.value_[self.apply(X)]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Most frequent class in the leaf of each sample; ties go to the
        smallest class."""
        return self.classes_[self.value_[self.apply(X)].argmax(axis=1)]


class ExtraTreesClassifier:
    """An ensemble of ExtraTree, all grown on the same training set.

    n_estimators:      number of trees (M >= 1).
    max_features:      K of every tree; None means round(sqrt(n_features)).
    min_samples_split: n_min of every tree.
    seed:              the trees get independent random streams derived from it.
    """

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, seed=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        """Grow n_estimators trees on (X, y).  Returns self."""
        X = np.asarray(X, dtype=float)
        k = self.max_features
        if k is None:
            k = max(1, round(np.sqrt(X.shape[1])))
        streams = np.random.SeedSequence(self.seed).spawn(self.n_estimators)
        self.trees_ = [ExtraTree(k, self.min_samples_split, s).fit(X, y) for s in streams]
        self.classes_ = self.trees_[0].classes_
        return self

    def predict_proba(self, X):
        """Fraction of the trees that vote for each class, shape (n, n_classes).

        Every tree casts exactly one vote per sample, for the class that
        ExtraTree.predict returns.  Each row therefore sums to 1.
        """
        X = np.asarray(X, dtype=float)
        votes = np.stack([t.value_[t.apply(X)].argmax(axis=1) for t in self.trees_])
        counts = np.zeros((len(X), len(self.classes_)))
        np.add.at(counts, (np.arange(len(X)), votes), 1)
        return counts / len(self.trees_)

    def predict(self, X):
        """Majority vote of the trees; ties go to the smallest class."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# Part 2
def variance_scores(go_left, y):
    y = np.asarray(y, dtype=float)
    go_left = np.asarray(go_left, dtype=bool)
    k = go_left.shape[1]
    if np.all(y == y[0]):
        return np.zeros(k)
    z = y - y.mean()
    s_left = z @ go_left
    n_left = go_left.sum(axis=0)
    n_right = len(y) - n_left
    between = (np.divide(s_left ** 2, n_left, out=np.zeros(k), where=n_left > 0)
               + np.divide(s_left ** 2, n_right, out=np.zeros(k), where=n_right > 0))
    return between / (z @ z)
