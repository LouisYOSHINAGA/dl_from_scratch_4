import numpy as np


if __name__ == "__main__":
    rng: np.random.Generator = np.random.default_rng(seed=0)

    """
    rewards: list[float] = []
    for n in range(1, 11):
        reward: float = rng.random()
        rewards.append(reward)
        Q: float = sum(rewards) / n
        print(Q)
    print()
    """

    Q: float = 0
    for n in range(1, 11):
        reward: float = rng.random()
        Q += (reward - Q) / n
        print(Q)
    print()