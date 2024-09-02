import numpy as np
import torch as t

a: np.ndarray = np.array([1, 2, 3])
b: np.ndarray = np.array([4, 5, 6])
a: t.Tensor = t.tensor(a)
b: t.Tensor = t.tensor(b)
c: t.Tensor= t.matmul(a, b)
print(f"{c=}")

a: t.Tensor = t.tensor([[1, 2], [3, 4]])
b: t.Tensor = t.tensor([[5, 6], [7, 8]])
c: t.Tensor = t.matmul(a, b)
print(f"{c=}")