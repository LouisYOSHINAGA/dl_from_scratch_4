import gym
import copy
import numpy as np
import dezero as dzr
import dezero.layers as L
import dezero.optimizers as O
import dezero.functions as F
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from typing import TypeAlias, Literal

State: TypeAlias = tuple[float, float, float, float]
States: TypeAlias = np.ndarray
Action: TypeAlias = Literal[0, 1]
Actions: TypeAlias = np.ndarray
Reward: TypeAlias = float
Rewards: TypeAlias = np.ndarray
DiscountRate: TypeAlias = float
Env: TypeAlias = gym.wrappers.time_limit.TimeLimit


class QNet(dzr.Model):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = L.Linear(in_size=in_size, out_size=hidden_size)
        self.l2 = L.Linear(in_size=hidden_size, out_size=hidden_size)
        self.l3 = L.Linear(in_size=hidden_size, out_size=out_size)

    def forward(self, x: dzr.Variable) -> dzr.Variable:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self, in_size: int =4, hidden_size: int =128, actions: list[Action] =[0, 1], gamma: DiscountRate =0.8,
                 epsilon: float =0.1, lr: float =0.0005, buffer_size: int =10000, batch_size: int =32) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.actions: list[Action] = actions
        self.gamma: DiscountRate = gamma
        self.epsilon: float = epsilon
        self.batch_size: int = batch_size

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.qnet = QNet(in_size=in_size, hidden_size=hidden_size, out_size=len(actions))
        self.qnet_target = QNet(in_size=in_size, hidden_size=hidden_size, out_size=len(actions))
        self.opt = O.Adam(lr).setup(self.qnet)

    def get_action(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        else:
            qs: dzr.Variable = self.qnet(state)  # (bs, actions)
            return qs.data.argmax()

    def update(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()

        next_qs: dzr.Variable = self.qnet_target(next_states).max(axis=1)
        next_qs.unchain()
        targets: dzr.Variable = rewards + (1 - dones) * self.gamma * next_qs

        qs: dzr.Variable = self.qnet(states)[range(self.batch_size), actions]
        loss: dzr.Variable = F.mean_squared_error(qs, targets)
        self.qnet.cleargrads()
        loss.backward()
        self.opt.update()

    def sync_qnet(self) -> None:
        self.qnet_target = copy.deepcopy(self.qnet)


def plot_rewards(rewards: list[Reward]) -> None:
    plt.figure(figsize=(8, 5))
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.plot(range(len(rewards)), rewards)
    plt.show()

def main(episodes: int =300, sync_interval: int =20, is_plot: bool =True, is_inference: bool =True) -> None:
    env: gym.wrappers.time_limit.TimeLimit = gym.make("CartPole-v0")
    agent = DQNAgent()

    reward_history: list[Reward] = []
    for episode in range(episodes):
        state: State = env.reset()
        done: bool = False
        total_reward: Reward = 0

        while not done:
            action: Action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()
        reward_history.append(total_reward)

    if is_plot:
        plot_rewards(reward_history)
    if is_inference:
        inference(env, agent)
    return reward_history

def inference(env: Env, agent: DQNAgent) -> None:
    agent.epsilon = 0
    state: State = env.reset()
    done: bool = False
    total_reward: Reward = 0

    while not done:
        action: Action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
    print(f"Total Reward: {total_reward}")

def main_avg() -> None:
    iters: int = 100
    episodes: int = 300
    reward_histories: np.ndarray = np.empty((iters, episodes))
    for i in range(iters):
        reward_histories[i] = main(episodes=episodes, is_plot=False, is_inference=False)
    plot_rewards(reward_histories.mean(axis=0))

if __name__ == "__main__":
    main()
    # main_avg()