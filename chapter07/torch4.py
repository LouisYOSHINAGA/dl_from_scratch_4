import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Any


class TwoLayerNet(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        y: t.Tensor = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

    @t.no_grad()
    def update(self, lr: float) -> None:
        for p in self.parameters():
            p -= lr * p.grad
            p.grad.zero_()


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


def inference(W1: t.Tensor, b1: t.Tensor, W2: t.Tensor, b2: t.Tensor,
              xrange: tuple[float, float] =(0, 1, 0.01)) -> tuple[t.Tensor, t.Tensor]:
    pred_x: t.Tensor = t.arange(*xrange).view(-1, 1)
    with t.no_grad():
        pred_y: t.Tensor = predict(pred_x, W1, b1, W2, b2)
    return pred_x, pred_y

def inference_model(model: nn.Module, xrange: tuple[float, float] =(0, 1, 0.01)) -> tuple[t.Tensor, t.Tensor]:
    pred_x: t.Tensor = t.arange(*xrange).view(-1, 1)
    with t.no_grad():
        pred_y: t.Tensor = model(pred_x)
    return pred_x, pred_y

def plot_data(data_x: t.Tensor, data_y: t.Tensor,
              pred_x: t.Tensor|None =None, pred_y: t.Tensor|None =None) -> None:
    plt.figure(figsize=(8, 6))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(data_x, data_y)
    if pred_x is not None and pred_y is not None:
        plt.plot(pred_x, pred_y, color=plt.get_cmap("tab10")(1))
    plt.tight_layout()
    plt.show()


def main() -> None:
    lr: float = 0.2
    iters: int = 10000

    x, y = gen_data()
    plot_data(x, y)  # fig. 7-10
    W1, b1, W2, b2 = init_weight(input=1, hidden=10, output=1)

    for i in range(iters):
        pred_y: t.Tensor = predict(x, W1, b1, W2, b2)
        loss: t.Tensor = F.mse_loss(pred_y, y)
        loss.backward()
        update(lr, W1, b1, W2, b2)

        if (i + 1) % 1000 == 0:
            print(f"[iter{i:04d}] loss = {loss.item():.5f}")

    pred_x, pred_y = inference(W1, b1, W2, b2)
    plot_data(x, y, pred_x, pred_y)  # fig. 7-12

def main_model() -> None:
    lr: float = 0.2
    iters: int = 10000

    x, y = gen_data()
    plot_data(x, y)  # fig. 7-10

    model = TwoLayerNet(in_size=1, hidden_size=10, out_size=1)

    for i in range(iters):
        pred_y: t.Tensor = model(x)
        loss: t.Tensor = F.mse_loss(pred_y, y)
        loss.backward()
        model.update(lr)

        if (i + 1) % 1000 == 0:
            print(f"[iter{i:04d}] loss = {loss.item():.5f}")

    pred_x, pred_y = inference_model(model)
    plot_data(x, y, pred_x, pred_y)  # fig. 7-12


if __name__ == "__main__":
    # main()
    main_model()