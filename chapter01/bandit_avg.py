import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent


if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilon = 0.1
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []
        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step+1))
        all_rates[run] = rates
    avg_rates = np.average(all_rates, axis=0)

    plt.xlabel("steps")
    plt.ylabel("rates")
    plt.plot(avg_rates)
    plt.show()


    epsilons = [0.1, 0.3, 0.01]
    rates_by_epsilon = []

    for epsilon in epsilons:
        runs = 200
        steps = 1000
        all_rates = np.zeros((runs, steps))

        for run in range(runs):
            bandit = Bandit()
            agent = Agent(epsilon)
            total_reward = 0
            rates = []
            for step in range(steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step+1))
            all_rates[run] = rates
        rates_by_epsilon.append(np.average(all_rates, axis=0))

    plt.xlabel("steps")
    plt.ylabel("rates")
    for epsilon, avg_rates in zip(epsilons, rates_by_epsilon):
        plt.plot(avg_rates, label=f"{epsilon}")
    plt.legend()
    plt.show()