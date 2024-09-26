import numpy as np
import dezero as dzr
import dezero.functions as F


a: np.ndarray = np.array([1, 2, 3])
b: np.ndarray = np.array([4, 5, 6])
c: dzr.Variable = F.matmul(a, b)
print(f"{c=}")

a: np.ndarray = np.array([[1, 2], [3, 4]])
b: np.ndarray = np.array([[5, 6], [7, 8]])
c: dzr.Variable = F.matmul(a, b)
print(f"{c=}")