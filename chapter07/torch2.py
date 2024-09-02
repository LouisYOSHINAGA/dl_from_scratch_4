import torch as t


def rosenbrock(x0: t.Tensor, x1: t.Tensor) -> t.Tensor:
    return 100 * (x1 - x0**2)**2 + (x0 - 1)**2


if __name__ == "__main__":
    x0: t.Tensor = t.tensor(0.0, requires_grad=True)
    x1: t.Tensor = t.tensor(2.0, requires_grad=True)

    lr: float = 0.001
    iters: int = 50000

    for i in range(iters):
        #print(f"{x0=}, {x1=}")

        y: t.Tensor = rosenbrock(x0, x1)
        y.backward()

        with t.no_grad():
            x0 -= lr * x0.grad
            x1 -= lr * x1.grad

        x0.grad.zero_()
        x1.grad.zero_()

    print(f"{x0=}, {x1=}")