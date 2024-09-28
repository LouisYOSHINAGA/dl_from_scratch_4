import numpy as np
from typing import Literal
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms: int =10) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.rates: float = self.rng.random(arms)

    def play(self, arm: int) -> Literal[0, 1]:
        return 1 if self.rng.random() < self.rates[arm] else 0


class Agent:
    def __init__(self, epsilon: float, action_size: int =10) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.epsilon: float = epsilon
        self.Qs: np.ndarray = np.zeros(action_size)
        self.ns: np.ndarray = np.zeros(action_size)

    def update(self, action: int, reward: float) -> None:
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        return self.rng.integers(0, len(self.Qs)) if self.rng.random() < self.epsilon \
               else np.argmax(self.Qs)


def main() -> None:
    """
    # rewards (0-th machine)
    bandit = Bandit()
    for i in range(3):
        print(bandit.play(0))
    """
    """
    # average of rewards (0-th machine)
    bandit = Bandit()
    Q: float = 0
    for n in range(1, 11):
        reward: Literal[0, 1] = bandit.play(0)
        Q += (reward - Q) / n
        print(f"estimated probability: {Q}")
    print(f"correct answer: {bandit.rates[0]}")
    """
    """
    # average of rewards
    arms: int = 10
    bandit = Bandit(arms)
    rng: np.random.Generator = np.random.default_rng()
    ns: np.ndarray = np.zeros(arms)
    Qs: np.ndarray = np.zeros(arms)
    for n in range(100):
        action: int = rng.integers(arms)
        reward: Literal[0, 1] = bandit.play(action)
        ns[action] += 1
        Qs[action] += (reward - Qs[action]) / ns[action]
        print(f"estimated probabilities: {Qs}")
    print(f"correct answers: {bandit.rates}")
    """

    steps: int = 1000

    bandit = Bandit()
    agent = Agent(epsilon=0.1)

    total_reward: int = 0
    total_rewards: list[int] = []
    rates: float = []

    for step in range(steps):
        action: int = agent.get_action()
        reward: Literal[0, 1] = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        total_rewards.append(total_reward)
        rates.append(total_reward / (step+1))
    print(f"{total_reward=}")

    plt.xlabel("steps")
    plt.ylabel("total reward")
    plt.plot(total_rewards)
    plt.show()

    plt.xlabel("steps")
    plt.ylabel("rates")
    plt.plot(rates)
    plt.show()


    # fig. 1-14
    runs: int = 10
    steps: int = 1000
    all_rates: np.ndarray = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon=0.1)

        total_reward: int = 0
        total_rewards: list[int] = []
        rates: float = []

        for step in range(steps):
            action: int = agent.get_action()
            reward: Literal[0, 1] = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            all_rates[run, step] = total_reward / (step+ 1)

    plt.xlabel("steps")
    plt.ylabel("rates")
    for rates in all_rates:
        plt.plot(rates)
    plt.show()


if __name__ == "__main__":
    main()