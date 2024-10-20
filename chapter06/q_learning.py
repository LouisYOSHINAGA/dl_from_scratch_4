if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld
from common.utils import greedy_probs

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]
Action: TypeAlias = Literal[0, 1, 2, 3]
Reward: TypeAlias = float
DiscountRate: TypeAlias = float
Policy: TypeAlias = dict[State, dict[Action, float]]
ActionValue: TypeAlias = float
ActionValueFunction: TypeAlias = dict[tuple[State, Action], ActionValue]


class QLearningAgent:
    def __init__(self):
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = 0.9
        self.alpha: float = 0.8
        self.epsilon: float = 0.1

        self.actions: list[Action] = [0, 1, 2, 3]
        self.pi: Policy = defaultdict(lambda: {action: 1/len(self.actions) for action in self.actions})
        self.b: Policy = defaultdict(lambda: {action: 1/len(self.actions) for action in self.actions})
        self.Q: ActionValueFunction = defaultdict(lambda: 0)

    def get_action(self, state) -> Action:
        action_probs: dict[Action, float] = self.b[state]
        return self.rng.choice(list(action_probs.keys()), p=list(action_probs.values()))

    def update(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> None:
        next_q_max: ActionValue = 0 if done else max([self.Q[next_state, action] for action in self.actions])
        target: float = reward + self.gamma * next_q_max
        self.Q[state, action] = self.Q[state, action] + self.alpha * (target - self.Q[state, action])
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


def main() -> None:
    env = GridWorld()
    agent = QLearningAgent()

    episodes: int = 10000
    for _ in range(episodes):
        state: State = env.reset()

        while True:
            action: Action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            if done:
                break
            state = next_state

    env.render_q(agent.Q)

if __name__ == "__main__":
    main()