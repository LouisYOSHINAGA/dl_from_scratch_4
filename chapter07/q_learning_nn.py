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
State: TypeAlias = tuple[int, int]|t.Tensor
Action: TypeAlias = Literal[0, 1, 2, 3]|t.Tensor
Reward: TypeAlias = float
DiscountRate: TypeAlias = float


class QNet(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1: nn.Module = nn.Linear(in_size, hidden_size)
        self.l2: nn.Module = nn.Linear(hidden_size, out_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class QLearningAgent:
    def __init__(self, state_size: int, hidden_size: int) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = 0.9
        self.lr: float = 0.01
        self.epsilon: float = 0.1

        self.actions: t.Tensor = t.Tensor([0, 1, 2, 3])
        self.action_size: int = len(self.actions)

        self.qnet = QNet(in_size=state_size, hidden_size=hidden_size, out_size=self.action_size)
        self.opt = optim.SGD(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return t.multinomial(self.actions, 1)
        else:
            qs: t.Tensor = self.qnet(state)
            return t.argmax(qs).item()

    def update(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> float:
        with t.no_grad():
            next_q: t.Tensor = t.zeros(1) if done else t.max(self.qnet(next_state), dim=1)[0]
        target = reward + self.gamma * next_q
        qs: t.Tensor = self.qnet(state)
        loss: t.Tensor = F.mse_loss(target, qs[:, action])

        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss.item()


def onehot(state: State, width: int, height: int) -> t.Tensor:
    x, y = state
    vec: t.Tensor = t.zeros(height*width, dtype=t.float32)
    vec[y * width + x] = 1
    return vec[np.newaxis, :]


def main() -> None:
    env = GridWorld()
    agent = QLearningAgent(env.width*env.height, hidden_size=100)

    episodes: int = 1000
    loss_history: list[float] = []
    for episode in range(episodes):
        total_loss: float = 0
        cnt: int = 0

        state: State = onehot(env.reset(), width=env.width, height=env.height)
        done: bool = False

        while not done:
            action: Action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = onehot(next_state, width=env.width, height=env.height)
            loss: float = agent.update(state, action, reward, next_state, done)

            total_loss += loss
            cnt += 1
            state = next_state

        loss_history.append(total_loss / cnt)
        if episode % 100 == 0:
            print(loss_history[-1])


if __name__ == "__main__":
    main()