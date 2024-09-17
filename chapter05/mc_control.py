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
ActionValueFunction: TypeAlias = dict[Action, float]  # Q(s,a)


class McAgent:
    def __init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = 0.9  # gamma
        self.actions: list[Action] = [0, 1, 2, 3]  # A
        self.action_size: int = len(self.actions)  # |A|

        self.pi: Policy = defaultdict(lambda: {action: 1/self.action_size for action in self.actions})
        self.Q: ActionValueFunction = defaultdict(lambda: 0)  # Q(s,a)

        self.alpha: float = 0.1
        self.epsilon: float = 0.1
        self.memory: list[tuple[State, Action, Reward]] = []

    def get_action(self, state: State) -> Action:
        action_probs: dict[Action, float] = self.pi[state]
        return self.rng.choice(list(action_probs.keys()), p=list(action_probs.values()))

    def add(self, state: State, action: Action, reward: Reward) -> None:
        self.memory.append((state, action, reward))

    def update(self) -> None:
        G: float = 0
        for state, action, reward in reversed(self.memory):
            G = reward + self.gamma * G
            self.Q[(state, action)] += self.alpha * (G - self.Q[(state, action)])
            self.pi[state] = greedy_probs(Q=self.Q, state=state, actions=self.actions, epsilon=self.epsilon)

    def reset(self) -> None:
        self.memory.clear()


def greedy_probs(Q: ActionValueFunction, state: State, actions: list[Action], epsilon: float =0) -> dict[Action, float]:
    max_action: Action = np.argmax([Q[(state, action)] for action in actions])
    action_probs: dict[Action, float] = {action: epsilon/len(actions) for action in actions}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


def main() -> None:
    env = GridWorld()
    agent = McAgent()

    episodes: int = 10000
    for _ in range(episodes):
        state: State = env.reset()
        agent.reset()

        while True:
            action: Action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add(state, action, reward)
            if done:
                agent.update()
                break
            state = next_state

    env.render_q(agent.Q)


if __name__ == "__main__":
    main()