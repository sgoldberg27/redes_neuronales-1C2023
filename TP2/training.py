from typing import Callable
import numpy as np
from tqdm import tqdm


seed = np.random.RandomState(seed=42)


CorrectionFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def initialize_weights(N, M):
    return seed.normal(0, 0.1, (N, M))


def oja_correction(W, X, Y):
    Z = np.dot(Y, W.T)
    return np.outer(X - Z, Y)


def sanger_correction(W, X, Y):
    M = W.shape[1]
    D = np.triu(np.ones((M, M)))
    Z = np.dot(W, Y.T * D)
    return (X.T - Z) * Y


def orthogonality(W):
    M = W.shape[1]
    return np.sum(np.abs(np.dot(W.T, W) - np.identity(M))) / 2


def PCA_train(
    X: np.ndarray,
    M: int,
    corr: CorrectionFunc,
    *,
    epochs: int = 1000,
    ort_threshold: float = 0.05,
    lr: float = 0.01,
) -> np.ndarray:
    W = initialize_weights(X.shape[1], M)

    for t in tqdm(range(epochs)):
        Y = np.dot(X, W)
        W += lr * corr(W, X, Y)

        if orthogonality(W) < ort_threshold:
            break

    return W
