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


class PolicyNet(dzr.Model):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = L.Linear(in_size=in_size, out_size=hidden_size)
        self.l2 = L.Linear(in_size=hidden_size, out_size=out_size)

    def forward(self, x: dzr.Variable) -> dzr.Variable:
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)
        return x


class ValueNet(dzr.Model):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = L.Linear(in_size=in_size, out_size=hidden_size)
        self.l2 = L.Linear(in_size=hidden_size, out_size=out_size)

    def forward(self, x: dzr.Variable) -> dzr.Variable:
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self, in_size: int, hidden_size: int, out_size_pi: int, out_size_v: int,
                 gamma: DiscountRate =0.98, lr_pi: float =0.0002, lr_v: float =0.0005) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = gamma

        self.pi = PolicyNet(in_size=in_size, hidden_size=hidden_size, out_size=out_size_pi)
        self.opt_pi = O.Adam(lr_pi).setup(self.pi)

        self.v = ValueNet(in_size=in_size, hidden_size=hidden_size, out_size=out_size_v)
        self.opt_v = O.Adam(lr_v).setup(self.v)

    def get_action(self, state: State) -> tuple[Action, dzr.Variable]:
        probs: dzr.Variable = self.pi(state[np.newaxis, :])[0]
        action: Action = self.rng.choice(len(probs), p=probs.data)
        return action, probs[action]

    def update(self, state: State, prob: dzr.Variable, reward: Reward, next_state: State, done: bool) -> None:
        target: dzr.Variable = reward + self.gamma * self.v(next_state[np.newaxis, :]) * (1 - done)
        target.unchain()

        v: dzr.Variable = self.v(state[np.newaxis, :])
        loss_v: dzr.Variable = F.mean_squared_error(v, target)

        delta: dzr.Variable = target - v
        delta.unchain()
        loss_pi: dzr.Varaible = - delta * F.log(prob)

        self.v.cleargrads()
        loss_v.backward()
        self.opt_v.update()

        self.pi.cleargrads()
        loss_pi.backward()
        self.opt_pi.update()


def main(is_plot: bool =True) -> list[Reward]:
    env: gym.wrappers.time_limit.TimeLimit = gym.make("CartPole-v0")
    agent = Agent(in_size=4, hidden_size=128, out_size_pi=2, out_size_v=1)

    reward_history: list[Reward] = []
    episodes: int = 3000
    for _ in range(episodes):
        state: State = env.reset()
        done: bool = False
        total_reward: Reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, prob, reward, next_state, done)
            state = next_state
            total_reward += reward
        reward_history.append(total_reward)

    if is_plot:
        plot_rewards(reward_history)  # fig. 9-11
    return reward_history

def plot_rewards(reward_history: list[Reward]) -> None:
    plt.figure(figsize=(7, 4))
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()

def main_avg() -> None:
    iters: int = 100
    episodes: int = 3000
    reward_histories: np.ndarray = np.empty((iters, episodes))
    for i in range(iters):
        reward_histories[i] = main()
    plot_rewards(reward_histories.mean(axis=0))  # fig. 9-11

if __name__ == "__main__":
    main()
    # main_avg()