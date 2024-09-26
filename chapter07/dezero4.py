import numpy as np
import dezero as dzr
import dezero.layers as L
import dezero.functions as F
import dezero.optimizers as optim
import matplotlib.pyplot as plt


class TwoLayerNet(dzr.Model):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = L.Linear(in_size=in_size, out_size=hidden_size)
        self.l2 = L.Linear(in_size=hidden_size, out_size=out_size)

    def forward(self, x: np.ndarray) -> dzr.Variable:
        y: dzr.Variable = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

    def update(self, lr: float) -> None:
        for p in self.params():
            p.data -= lr * p.grad.data


def gen_data(seed: int =0) -> tuple[np.ndarray, np.ndarray]:
    rng: np.random.Generator = np.random.default_rng(seed)
    x: np.ndarray = rng.random((100, 1))
    y: np.ndarray = np.sin(2 * np.pi * x) + rng.random((100, 1))
    return x, y

def init_weight(input: int, hidden: int, output: int, seed: int =0) -> tuple[dzr.Variable, dzr.Variable, dzr.Variable, dzr.Variable]:
    rng: np.random.Generator = np.random.default_rng(seed)
    W1: dzr.Variable = dzr.Variable(0.01 * rng.standard_normal((input, hidden)))
    b1: dzr.Variable = dzr.Variable(np.zeros(hidden))
    W2: dzr.Variable = dzr.Variable(0.01 * rng.standard_normal((hidden, output)))
    b2: dzr.Variable = dzr.Variable(np.zeros(output))
    return W1, b1, W2, b2

def predict(x: np.ndarray, W1: dzr.Variable, b1: dzr.Variable, W2: dzr.Variable, b2: dzr.Variable) -> dzr.Variable:
    y: dzr.Variable = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

def cleargrad(*args: tuple[dzr.Variable]) -> None:
    for arg in args:
        arg.cleargrad()

def update(lr: float, *args: tuple[dzr.Variable]) -> None:
    for arg in args:
        arg.data -= lr * arg.grad.data


def inference(W1: dzr.Variable, b1: dzr.Variable, W2: dzr.Variable, b2: dzr.Variable,
              xrange: tuple[float, float] =(0, 1, 0.01)) -> tuple[np.ndarray, dzr.Variable]:
    pred_x: np.ndarray= np.arange(*xrange)[:, np.newaxis]
    pred_y: dzr.Variable = predict(pred_x, W1, b1, W2, b2)
    return pred_x, pred_y

def inference_model(model: dzr.Model, xrange: tuple[float, float] =(0, 1, 0.01)) -> tuple[np.ndarray, dzr.Variable]:
    pred_x: np.ndarray = np.arange(*xrange)[:, np.newaxis]
    pred_y: dzr.Variable = model(pred_x)
    return pred_x, pred_y

def plot_data(data_x: np.ndarray, data_y: np.ndarray,
              pred_x: np.ndarray|None =None, pred_y: dzr.Variable|None =None) -> None:
    plt.figure(figsize=(8, 6))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(data_x, data_y)
    if pred_x is not None and pred_y is not None:
        plt.plot(pred_x[:, 0], pred_y[:, 0].data, color=plt.get_cmap("tab10")(1))
    plt.tight_layout()
    plt.show()


def main() -> None:
    lr: float = 0.2
    iters: int = 10000

    x, y = gen_data()
    plot_data(x, y)  # fig. 7-10
    W1, b1, W2, b2 = init_weight(input=1, hidden=10, output=1)

    for i in range(iters):
        pred_y: dzr.Variable = predict(x, W1, b1, W2, b2)
        loss: dzr.Variable = F.mean_squared_error(pred_y, y)

        cleargrad(W1, b1, W2, b2)
        loss.backward()
        update(lr, W1, b1, W2, b2)

        if (i + 1) % 1000 == 0:
            print(f"[iter{i:04d}] loss = {loss.data:.5f}")

    pred_x, pred_y = inference(W1, b1, W2, b2)
    plot_data(x, y, pred_x, pred_y)  # fig. 7-12

def main_model() -> None:
    lr: float = 0.2
    iters: int = 10000

    x, y = gen_data()
    plot_data(x, y)  # fig. 7-10

    model = TwoLayerNet(in_size=1, hidden_size=10, out_size=1)

    for i in range(iters):
        pred_y: dzr.Variable = model(x)
        loss: dzr.Variable = F.mean_squared_error(pred_y, y)

        model.cleargrads()
        loss.backward()
        model.update(lr)

        if (i + 1) % 1000 == 0:
            print(f"[iter{i:04d}] loss = {loss.data:.5f}")

    pred_x, pred_y = inference_model(model)
    plot_data(x, y, pred_x, pred_y)  # fig. 7-12

def main_opt() -> None:
    lr: float = 0.2
    iters: int = 10000

    x, y = gen_data()
    plot_data(x, y)  # fig. 7-10

    model = TwoLayerNet(in_size=1, hidden_size=10, out_size=1)
    opt = optim.SGD(lr=lr)
    opt.setup(model)

    for i in range(iters):
        pred_y: dzr.Variable = model(x)
        loss: dzr.Variable = F.mean_squared_error(pred_y, y)

        model.cleargrads()
        loss.backward()
        opt.update()

        if (i + 1) % 1000 == 0:
            print(f"[iter{i:04d}] loss = {loss.data:.5f}")

    pred_x, pred_y = inference_model(model)
    plot_data(x, y, pred_x, pred_y)  # fig. 7-12


if __name__ == "__main__":
    # main()
    # main_model()
    main_opt()