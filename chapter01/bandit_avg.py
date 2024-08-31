import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from bandit import Bandit, Agent


if __name__ == "__main__":
    runs: int = 200
    steps: int = 1000
    all_rates: np.ndarray = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon=0.1)
        total_reward: int = 0
        rates = []
        for step in range(steps):
            action: int = agent.get_action()
            reward: Literal[0, 1] = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step+1))
        all_rates[run] = rates
    avg_rates: np.ndarray = np.average(all_rates, axis=0)  # (step, )

    plt.xlabel("steps")
    plt.ylabel("rates")
    plt.plot(avg_rates)
    plt.show()


    # fig. 1-17
    epsilons: list[float] = [0.1, 0.3, 0.01]
    runs: int = 200
    steps: int = 1000
    all_rates_by_epsilon: np.ndarray = np.zeros((len(epsilons), runs, steps))

    for i, epsilon in enumerate(epsilons):
        all_rates: np.ndarray = np.zeros((runs, steps))
        for run in range(runs):
            bandit = Bandit()
            agent = Agent(epsilon=epsilon)
            total_reward: int = 0
            rates: float = []
            for step in range(steps):
                action: int = agent.get_action()
                reward: Literal[0, 1] = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step+1))
            all_rates[run] = rates
        all_rates_by_epsilon[i] = np.average(all_rates, axis=0)

    plt.xlabel("steps")
    plt.ylabel("rates")
    for epsilon, avg_rates in zip(epsilons, all_rates_by_epsilon):
        plt.plot(avg_rates, label=f"{epsilon}")
    plt.legend()
    plt.show()