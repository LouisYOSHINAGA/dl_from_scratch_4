import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from bandit import Agent


class NonStatBandit:
    def __init__(self, arms: int =10) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.arms: int = arms
        self.rates: float = self.rng.random(arms)

    def play(self, arm: int) -> Literal[0, 1]:
        rate: Literal[0, 1] = self.rates[arm]
        self.rates += 0.1 * self.rng.standard_normal(self.arms)
        return 1 if self.rng.random() < rate else 0


class AlphaAgent:
    def __init__(self, epsilon: float, alpha: float, actions: int =10) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.epsilon: float = epsilon
        self.Qs: np.ndarray = np.zeros(actions)
        self.alpha: float = alpha

    def update(self, action: int, reward: Literal[0, 1]) -> None:
        self.Qs[action] += self.alpha * (reward - self.Qs[action])

    def get_action(self) -> int:
        return self.rng.integers(len(self.Qs)) if self.rng.random() < self.epsilon \
               else np.argmax(self.Qs)


def main() -> None:
    runs: int = 200
    steps: int = 1000
    epsilon: float = 0.1
    alpha: float = 0.8

    results: dict[str, np.ndarray] = {}
    for agent_type in ["sample average", "alpha const average"]:
        all_rates: np.ndarray = np.zeros((runs, steps))
        for run in range(runs):
            bandit = NonStatBandit()
            if agent_type == "sample average":
                agent = Agent(epsilon)
            elif agent_type == "alpha const average":
                agent = AlphaAgent(epsilon, alpha)

            total_reward: int = 0
            rates: list[float] = []
            for step in range(steps):
                action: int = agent.get_action()
                reward: Literal[0, 1] = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step+1))
            all_rates[run] = rates
        avg_rates = np.average(all_rates, axis=0)
        results[agent_type] = avg_rates

    plt.xlabel("steps")
    plt.ylabel("rates")
    for agent_type, avg_rates in results.items():
        plt.plot(avg_rates, label=agent_type)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()