if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld
from common.utils import greedy_probs

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]  # s
Action: TypeAlias = Literal[0, 1, 2, 3]  # a
Policy: TypeAlias = dict[State, dict[Action, float]]  # pi(a|s)
Reward: TypeAlias = float  # r
DiscountRate: TypeAlias = float  # gamma
ActionValueFunction: TypeAlias = dict[Action, float]  # Q(s,a)


class McOffPolicyAgent:
    def __init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = 0.9  # gamma
        self.actions: list[Action] = [0, 1, 2, 3]  # A
        self.action_size = len(self.actions)  # |A|

        self.pi: Policy = defaultdict(lambda: {action: 1/self.action_size for action in self.actions})  # pi(a|s)
        self.b: Policy = defaultdict(lambda: {action: 1/self.action_size for action in self.actions})  # b(a|s)
        self.Q: ActionValueFunction  = defaultdict(lambda: 0)  # Q(s,a)

        self.epsilon: float = 0.1
        self.alpha: float = 0.2
        self.memory: list[tuple[State, Action, Reward]] = []

    def get_action(self, state: State) -> Action:
        action_probs: dict[Action, float] = self.b[state]
        return self.rng.choice(list(action_probs.keys()), p=list(action_probs.values()))

    def add(self, state: State, action: Action, reward: Reward) -> None:
        self.memory.append((state, action, reward))

    def update(self) -> None:
        G: float = 0
        rho: float = 1
        for state, action, reward in reversed(self.memory):
            G = reward + rho * self.gamma * G
            self.Q[(state, action)] += self.alpha * (G - self.Q[(state, action)])
            rho *= self.pi[state][action] / self.b[state][action]  # rho = \prod_{t} pi(A_t|S_t)/b(A_t|S_t)
            self.pi[state] = greedy_probs(Q=self.Q, state=state, epsilon=0)
            self.b[state] = greedy_probs(Q=self.Q, state=state, epsilon=self.epsilon)

    def reset(self) -> None:
        self.memory.clear()


def main() -> None:
    env = GridWorld()
    agent = McOffPolicyAgent()

    episodes = 10000
    for _ in range(episodes):
        state = env.reset()
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