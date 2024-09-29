if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import dezero as dzr
import dezero.functions as F
import dezero.layers as L
import dezero.optimizers as O
from common.gridworld import GridWorld

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]
Action: TypeAlias = Literal[0, 1, 2, 3]
Reward: TypeAlias = float
DiscountRate: TypeAlias = float


class QNet(dzr.Model):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = L.Linear(in_size=in_size, out_size=hidden_size)
        self.l2 = L.Linear(in_size=hidden_size, out_size=out_size)

    def forward(self, x: np.ndarray) -> dzr.Variable:
        y: dzr.Variable = F.relu(self.l1(x))
        y = self.l2(y)
        return y


class QLearningAgent:
    def __init__(self, width: int, height: int, actions: list[Action], hidden_size: int) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = 0.9
        self.lr: float = 0.01
        self.epsilon: float = 0.1

        self.width: int = width
        self.height: int = height
        self.actions: list[Action] = actions

        self.qnet = QNet(in_size=width*height, hidden_size=hidden_size, out_size=len(actions))
        self.opt = O.SGD(self.lr)
        self.opt.setup(self.qnet)

    def onehot(self, state: State) -> np.ndarray:
        h, w = state
        vec: np.ndarray = np.zeros(self.width*self.height, dtype=np.float32)
        vec[h * self.width + w] = 1
        return vec[np.newaxis, :]

    def get_action(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        else:
            qs: dzr.Variable = self.qnet(self.onehot(state))
            return qs.data.argmax()

    def update(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> float:
        next_q: dzr.Variable = self.qnet(self.onehot(next_state)).max(axis=1)
        next_q.unchain()
        target: dzr.Variable = reward + (1 - int(done)) * self.gamma * next_q

        q: Action = self.qnet(self.onehot(state))[:, action]
        loss: dzr.Variable = F.mean_squared_error(target, q)
        self.qnet.cleargrads()
        loss.backward()
        self.opt.update()
        return loss.data


def main() -> None:
    env = GridWorld()
    agent = QLearningAgent(width=env.width, height=env.height, actions=[0, 1, 2, 3], hidden_size=100)

    episodes: int = 1000
    loss_history: list[float] = []
    for episode in range(episodes):
        total_loss: float = 0
        cnt: int = 0

        state: State = env.reset()
        done: bool = False

        while not done:
            action: Action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            loss: float = agent.update(state, action, reward, next_state, done)

            total_loss += loss
            cnt += 1
            state = next_state

        loss_history.append(total_loss / cnt)
        if episode % 100 == 0:
            print(loss_history[-1])


if __name__ == "__main__":
    main()