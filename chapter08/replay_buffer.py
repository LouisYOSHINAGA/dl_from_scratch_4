import random
import numpy as np
from collections import deque
from typing import TypeAlias, Literal, Deque

State: TypeAlias = tuple[float, float, float, float]
States: TypeAlias = np.ndarray
Action: TypeAlias = Literal[0, 1]
Actions: TypeAlias = np.ndarray
Reward: TypeAlias = float
Rewards: TypeAlias = np.ndarray
Data: TypeAlias = tuple[State, Action, Reward, State, bool]
Datas: TypeAlias = tuple[States, Actions, Rewards, States, np.ndarray]


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.buffer: Deque[Data] = deque(maxlen=buffer_size)
        self.batch_size: int = batch_size

    def add(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def get_batch(self) -> Datas:
        data: list[Data] = random.sample(self.buffer, self.batch_size)
        states: States = np.stack([x[0] for x in data])
        actions: Actions = np.array([x[1] for x in data])
        rewards: Rewards = np.array([x[2] for x in data])
        next_states: States = np.stack([x[3] for x in data])
        dones: np.ndarray = np.array([x[4] for x in data]).astype(np.int32)
        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    import gym

    env: gym.wrappers.time_limit.TimeLimit = gym.make("CartPole-v0")
    buf = ReplayBuffer(buffer_size=10000, batch_size=32)

    for episode in range(10):
        state: State = env.reset()
        action: Action = 0  # fix
        done: bool = False

        while not done:
            next_state, reward, done, info = env.step(action)
            buf.add(state, action, reward, next_state, done)
            state = next_state

    state, action, reward, next_state, done = buf.get_batch()
    print(f"{len(buf)=}")
    print(f"{state.shape=}")
    print(f"{action.shape=}")
    print(f"{reward.shape=}")
    print(f"{next_state.shape=}")
    print(f"{done.shape=}")