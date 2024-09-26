import numpy as np
import dezero as dzr
import dezero.functions as F
import matplotlib.pyplot as plt


def predict(x: dzr.Variable, W: dzr.Variable, b: dzr.Variable) -> dzr.Variable:
    return F.matmul(x, W) + b

def mean_squared_error(x0: dzr.Variable, x1: dzr.Variable) -> dzr.Variable:
    diff: dzr.Variable = x0 - x1
    return F.sum(diff ** 2) / len(diff)

def plot_regression(x: np.ndarray, y: np.ndarray, W: dzr.Variable|None =None, b: dzr.Variable|float =0) -> None:
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, marker='.', color=plt.get_cmap("tab10")(0))
    if W is not None:
        reg_x: np.ndarray = np.linspace(min(x), max(x), 1000)
        reg_y: dzr.Variable = predict(reg_x, W, b)
        plt.plot(reg_x[:, 0], reg_y[:, 0].data, color=plt.get_cmap("tab10")(1))
    plt.show()


def main() -> None:
    rng: np.random.Generator = np.random.default_rng(0)

    x: np.ndarray = rng.random((100, 1))
    y: np.ndarray = 5 + 2 * x + rng.random((100, 1))  # y = b + W * x + noise
    plot_regression(x, y)  # fig. 7-5

    W: dzr.Variable = dzr.Variable(np.zeros((1, 1)))
    b: dzr.Variable = dzr.Variable(np.zeros(1))

    lr: float = 0.1
    iters: int = 100

    for i in range(iters):
        y_pred: dzr.Variable = predict(x, W, b)
        loss: dzr.Variable = mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data

        if i % 10 == 0:
            print(loss.data)

    print(f"====================")
    print(f"W = {W.data}")
    print(f"b = {b.data}")
    plot_regression(x, y, W, b)  # fig. 7-9


if __name__ == "__main__":
    main()