import numpy as np
import torch as t
import torch.nn.functional as F
from typing import Any


def gen_data() -> tuple[t.Tensor, t.Tensor]:
    x: t.Tensor = t.rand(100, 1)
    y: t.Tensor = t.sin(2 * t.pi * x) + t.rand(100, 1)
    return x, y

def init_weight(input: int, hidden: int, output: int) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    rng: np.random.Generator = np.random.default_rng()
    W1: np.ndarray|t.Tensor = 0.01 * rng.random((hidden, input), dtype=np.float32)
    W1 = t.tensor(W1, requires_grad=True)
    b1: t.Tensor = t.zeros(hidden, requires_grad=True)
    W2: np.ndarray|t.Tensor = 0.01 * rng.random((output, hidden), dtype=np.float32)
    W2 = t.tensor(W2, requires_grad=True)
    b2: t.Tensor = t.zeros(output, requires_grad=True)
    return W1, b1, W2, b2

def predict(x: t.Tensor, W1: t.Tensor, b1: t.Tensor, W2: t.Tensor, b2: t.Tensor) -> t.Tensor:
    x = F.linear(x, W1, b1)
    x = F.sigmoid(x)
    x = F.linear(x, W2, b2)
    return x

def update(lr: float, *args: tuple[Any]) -> None:
    with t.no_grad():
        for arg in args:
            arg -= lr * arg.grad
            arg.grad.zero_()

def main() -> None:
    lr: float = 0.2
    iters: int = 10000

    x, y = gen_data()
    W1, b1, W2, b2 = init_weight(input=1, hidden=10, output=1)

    for i in range(iters):
        y_pred: t.Tensor = predict(x, W1, b1, W2, b2)
        loss: t.Tensor = F.mse_loss(y_pred, y)
        loss.backward()
        update(lr, W1, b1, W2, b2)

        if (i + 1) % 1000 == 0:
            print(f"[iter{i:04d}] loss = {loss.item():.5f}")


if __name__ == "__main__":
    main()