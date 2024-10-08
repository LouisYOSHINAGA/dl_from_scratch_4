if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]
Action: TypeAlias = Literal[0, 1, 2, 3]
Policy: TypeAlias = dict[State, dict[Action, float]]
Reward: TypeAlias = float
DiscountRate: TypeAlias = float
StateValueFunction: TypeAlias = dict[State, float]


class TdAgent:
    def __init__(self) -> None:
        self.gamma: DiscountRate = 0.9
        self.alpha: float = 0.01

        self.actions: list[Action] = [0, 1, 2, 3]
        self.pi: Policy = defaultdict(lambda: {action: 1/len(self.actions) for action in self.actions})
        self.V: StateValueFunction = defaultdict(lambda: 0)

    def get_action(self, state: State) -> Action:
        action_probs: dict[Action, float] = self.pi[state]
        return np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))

    def eval(self, state: State, reward: Reward, next_state: State, done: bool) -> None:
        next_V: Reward = 0 if done else self.V[next_state]
        self.V[state] = self.V[state] + self.alpha * (reward + self.gamma * next_V - self.V[state])


def main() -> None:
    env = GridWorld()
    agent = TdAgent()

    episodes = 1000
    for _ in range(episodes):
        state: State = env.reset()

        while True:
            action: Action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.eval(state, reward, next_state, done)
            if done:
                break
            state = next_state

    env.render_v(agent.V)

if __name__ == "__main__":
    main()