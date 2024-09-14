import torch as t
import matplotlib.pyplot as plt


def predict(x: t.Tensor, W: t.Tensor, b: t.Tensor) -> t.Tensor:
    return b + W * x

def mean_squared_error(x0: t.Tensor, x1: t.Tensor) -> t.Tensor:
    diff: t.Tensor = x0 - x1
    return t.sum(diff ** 2) / len(diff)

def plot_regression(x: t.Tensor, y: t.Tensor, W: t.Tensor|None =None, b: t.Tensor|float =0) -> None:
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, marker='.', color=plt.get_cmap("tab10")(0))
    if W is not None:
        with t.no_grad():
            reg_x: t.Tensor = t.linspace(t.min(x), t.max(x), 1000)
            reg_y: t.Tensor = predict(reg_x, W, b)[0]
            plt.plot(reg_x, reg_y, color=plt.get_cmap("tab10")(1))
    plt.show()

def main() -> None:
    x: t.Tensor = t.rand(100, 1)
    y: t.Tensor = 5 + 2 * x + t.rand(100, 1)  # y = b + W * x + noise
    plot_regression(x, y)  # fig. 7-5

    W: t.Tensor = t.zeros(1, 1, requires_grad=True)
    b: t.Tensor = t.zeros(1, requires_grad=True)

    lr: float = 0.1
    iters: int = 100

    for i in range(iters):
        y_pred: t.Tensor = predict(x, W, b)
        loss: t.Tensor = mean_squared_error(y, y_pred)
        loss.backward()

        with t.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad

        W.grad.zero_()
        b.grad.zero_()

        if i % 10 == 0:
            print(loss.item())

    print(f"====================")
    print(f"W = {W.item()}")
    print(f"b = {b.item()}")
    plot_regression(x, y, W, b)  # fig. 7-9


if __name__ == "__main__":
    main()