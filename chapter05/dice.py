import numpy as np


def sample(dices: int =2) -> int:
    rng: np.random.Generator = np.random.default_rng()
    x: int = 0
    for _ in range(dices):
        x += rng.choice([1, 2, 3, 4, 5, 6])
    return x


if __name__ == "__main__":
    print(sample())
    print(sample())
    print(sample())

    trial: int = 1000
    V: float = 0
    n: int = 0
    for _ in range(trial):
        s: int = sample()
        n += 1
        V += (s - V) / n
    print(V)