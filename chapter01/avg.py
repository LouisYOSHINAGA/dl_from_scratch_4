import numpy as np


if __name__ == "__main__":
    np.random.seed(0)

    rewards = []
    for n in range(1, 11):
        reward = np.random.rand()
        rewards.append(reward)
        Q = sum(rewards) / n
        print(Q)
    print()


    np.random.seed(0)

    Q = 0
    for n in range(1, 11):
        reward = np.random.rand()
        Q += (reward - Q) / n
        print(Q)
    print()