if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict, deque
from common.utils import greedy_probs
from common.gridworld import GridWorld

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]
Action: TypeAlias = Literal[0, 1, 2, 3]
Reward: TypeAlias = float
DiscountRate: TypeAlias = float
Policy: TypeAlias = dict[State, dict[Action, float]]
ActionValue: TypeAlias = float
ActionValueFunction: TypeAlias = dict[tuple[State, Action], ActionValue]


class SarsaOffPolicyAgent:
    def __init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.gamma: DiscountRate = 0.9
        self.alpha: float = 0.8
        self.epsilon: float = 0.1

        self.actions: list[Action] = [0, 1, 2, 3]
        self.pi: Policy = defaultdict(lambda: {action: 1/len(self.actions) for action in self.actions})
        self.b: Policy = defaultdict(lambda: {action: 1/len(self.actions) for action in self.actions})
        self.Q: ActionValueFunction = defaultdict(lambda: 0)
        self.memory: list[tuple[State, Action, Reward, bool]] = deque(maxlen=2)

    def reset(self) -> None:
        self.memory.clear()

    def get_action(self, state: State) -> Action:
        action_probs: dict[Action, float] = self.b[state]
        return self.rng.choice(list(action_probs.keys()), p=list(action_probs.values()))

    def update(self, state: State, action: Action, reward: Reward, done: bool) -> None:
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        next_q: ActionValue = 0 if done else self.Q[next_state, next_action]
        rho: float = 1 if done else self.pi[next_state][next_action] / self.b[next_state][next_action]

        self.Q[state, action] = self.Q[state, action] \
                              + self.alpha * (rho * (reward + self.gamma * next_q) - self.Q[state, action])
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


def main() -> None:
    env = GridWorld()
    agent = SarsaOffPolicyAgent()

    episodes = 10000
    for _ in range(episodes):
        state: State = env.reset()
        agent.reset()

        while True:
            action: Action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, done)

            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state

    env.render_q(agent.Q)

if __name__ == "__main__":
    main()