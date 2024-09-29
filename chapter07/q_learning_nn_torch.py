if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.gridworld import GridWorld

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]
StateOneHot: TypeAlias = t.Tensor
Action: TypeAlias = Literal[0, 1, 2, 3]
Reward: TypeAlias = float
DiscountRate: TypeAlias = float


class QNet(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1: nn.Module = nn.Linear(in_size, hidden_size)
        self.l2: nn.Module = nn.Linear(hidden_size, out_size)

    def forward(self, x: StateOneHot) -> t.Tensor:
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class QLearningAgent:
    def __init__(self, width: int, height: int, actions: list[Action], hidden_size: int) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = 0.9
        self.lr: float = 0.001
        self.epsilon: float = 0.1

        self.width: int = width
        self.height: int = height
        self.actions: t.Tensor = t.tensor(actions, dtype=t.float32)

        self.qnet = QNet(in_size=width*height, hidden_size=hidden_size, out_size=len(self.actions))
        self.opt = optim.SGD(self.qnet.parameters(), lr=self.lr)

    def onehot(self, state: State) -> StateOneHot:
        h, w = state
        vec: t.Tensor = t.zeros(self.width*self.height, dtype=t.float32)
        vec[h * self.width + w] = 1
        return vec.unsqueeze(dim=0)

    def get_action(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return t.multinomial(self.actions, 1).item()
        else:
            qs: t.Tensor = self.qnet(self.onehot(state))
            return qs.argmax().item()

    def update(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> float:
        with t.no_grad():
            next_q: t.Tensor = t.max(self.qnet(self.onehot(next_state)), dim=1)[0]
        target: t.Tensor = reward + (1 - int(done)) * self.gamma * next_q
        qs: t.Tensor = self.qnet(self.onehot(state))
        loss: t.Tensor = F.mse_loss(qs[:, action], target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


def main() -> None:
    env = GridWorld()
    agent = QLearningAgent(width=env.width, height=env.height, actions=[0, 1, 2, 3], hidden_size=100)

    episodes: int = 100
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
        if episode % 10 == 0:
            print(loss_history[-1])


if __name__ == "__main__":
    main()