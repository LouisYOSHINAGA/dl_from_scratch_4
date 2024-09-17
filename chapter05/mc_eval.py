if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]  # s
Action: TypeAlias = Literal[0, 1, 2, 3]  # a
Policy: TypeAlias = dict[State, dict[Action, float]]  # pi(a|s)
Reward: TypeAlias = float  # r
DiscountRate: TypeAlias = float  # gamma
StateValueFunction: TypeAlias = dict[State, float]  # v(s)


class RandomAgent:
    def __init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = 0.9  # gamma
        self.actions: list[Action] = [0, 1, 2, 3]  # A
        self.action_size: int = len(self.actions)  # |A|

        self.pi: Policy = defaultdict(lambda: {action: 1/self.action_size for action in self.actions})  # pi(a|s)
        self.V: StateValueFunction = defaultdict(lambda: 0)  # V(s)

        self.cnts: dict[State, int] = defaultdict(lambda: 1)
        self.memory: list[tuple[State, Action, Reward]] = []

    def get_action(self, state: State) -> Action:
        action_probs: dict[Action, float] = self.pi[state]
        return self.rng.choice(list(action_probs.keys()), p=list(action_probs.values()))

    def add(self, state: State, action: Action, reward: Reward) -> None:
        self.memory.append((state, action, reward))

    def eval(self) -> None:
        G: float = 0
        for state, _, reward in reversed(self.memory):
            G = reward + self.gamma * G  # G_{s} = R_{t} + gamma * R_{t+1} + ... = R_{t} + gamma * G_{s'}
            self.V[state] += (G - self.V[state]) / self.cnts[state]  # V_{n} = V_{n-1} + (s_{n} - V_{n-1}) / n
            self.cnts[state] += 1

    def reset(self) -> None:
        self.memory.clear()


def main() -> None:
    env = GridWorld()
    agent = RandomAgent()

    episodes: int = 1000
    for _ in range(episodes):
        state: State = env.reset()
        agent.reset()

        while True:
            action: Action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add(state, action, reward)
            if done:
                agent.eval()
                break
            state = next_state
    env.render_v(agent.V)


if __name__ == "__main__":
    main()