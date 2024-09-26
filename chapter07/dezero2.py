import numpy as np
import dezero as dzr


def rosenbrock(x0: dzr.Variable, x1: dzr.Variable) -> dzr.Variable:
    return 100 * (x1 - x0 ** 2) **2 + (x0 - 1) ** 2


def main() -> None:
    x0: dzr.Variable = dzr.Variable(np.array(0.0))
    x1: dzr.Variable = dzr.Variable(np.array(2.0))

    lr: float = 0.001
    iters: int = 10000

    for i in range(iters):
        # print(f"{x0=}, {x1=}")
        y: dzr.Variable = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad.data
        x1.data -= lr * x1.grad.data

    print(f"{x0=}, {x1=}")


if __name__ == "__main__":
    main()