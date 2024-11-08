import numpy as np
import gym
import dezero as dzr
import dezero.layers as L
import dezero.optimizers as O
import dezero.functions as F
import matplotlib.pyplot as plt
from typing import TypeAlias, Literal

State: TypeAlias = tuple[float, float, float, float]
Action: TypeAlias = Literal[0, 1]
Reward: TypeAlias = float
DiscountRate: TypeAlias = float


class Policy(dzr.Model):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = L.Linear(in_size=in_size, out_size=hidden_size)
        self.l2 = L.Linear(in_size=hidden_size, out_size=out_size)

    def forward(self, x: dzr.Variable) -> dzr.Variable:
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    def __init__(self, in_size: int, hidden_size: int, out_size: int, gamma: DiscountRate =0.98, lr: float =0.0002) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = gamma
        self.lr: float = lr

        self.memory: list[tuple[Reward, float]] = []
        self.pi = Policy(in_size=in_size, hidden_size=hidden_size, out_size=out_size)
        self.opt = O.Adam(lr).setup(self.pi)

    def get_action(self, state: State) -> tuple[Action, float]:
        probs: dzr.Variable = self.pi(state[np.newaxis, :])[0]
        action: Action = self.rng.choice(len(probs), p=probs.data)
        return action, probs[action]

    def add(self, reward: Reward, prob: float) -> None:
        self.memory.append((reward, prob))

    def update(self) -> None:
        self.pi.cleargrads()

        G: float = 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G

        loss: float = 0
        for reward, prob in self.memory:
            loss -= G * F.log(prob)
        loss.backward()

        self.opt.update()
        self.memory = []


def main(is_plot: bool =True) -> list[Reward]:
    env: gym.wrappers.time_limit.TimeLimit = gym.make("CartPole-v0")
    agent = Agent(in_size=4, hidden_size=128, out_size=2)

    reward_history: list[Reward] = []
    episodes: int = 3000
    for _ in range(episodes):
        state: State = env.reset()
        done: bool = False
        total_reward: Reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        agent.update()
        reward_history.append(total_reward)

    if is_plot:
        plot_rewards(reward_history)  # fig. 9-2
    return reward_history

def plot_rewards(reward_history: list[Reward]) -> None:
    plt.figure(figsize=(7, 4))
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()

def main_avg() -> None:
    iters: int = 3
    episodes: int = 3000
    reward_histories: np.ndarray = np.empty((iters, episodes))
    for i in range(iters):
        reward_histories[i] = main()
    plot_rewards(reward_histories.mean(axis=0))  # fig. 9-3

if __name__ == "__main__":
    main()
    # main_avg()