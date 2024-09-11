if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from common.gridworld import GridWorld
from policy_iter import greedy_policy

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]
Action: TypeAlias = Literal[0, 1, 2, 3]
StateValueFunction: TypeAlias = tuple[State, float]  # v(s)
DiscountRate: TypeAlias = float  # gamma
Policy: TypeAlias = dict[State, tuple[Action, float]]  # pi(a|s)


def value_iter_onestep(V: StateValueFunction, env: GridWorld, gamma: DiscountRate) -> StateValueFunction:
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values: dict[Action, float] = {}  # {a: r(s,a,s') + gamma * v(s'))
        for action in env.actions():
            next_state: State = env.next_state(state, action)  # s'
            action_values[action] = env.reward(state, action, next_state) + gamma * V[next_state]  # r(s,a,s') + gamma * v(s')
        V[state] = max(action_values.values())
    return V

def value_iter(V: StateValueFunction, env: GridWorld, gamma: DiscountRate,
               threshold: float =0.001, is_render: bool =True) -> StateValueFunction:
    while True:
        if is_render:
            env.render_v(V)

        old_V: StateValueFunction = V.copy()  # V_{k}
        V = value_iter_onestep(V, env, gamma)  # V_{k+1}

        if max([V[state] - old_V[state] for state in V.keys()]) < threshold:
            break
    return V


if __name__ == "__main__":
    env = GridWorld()
    gamma: DiscountRate = 0.9
    V: StateValueFunction = defaultdict(lambda: 0)
    V = value_iter(V, env, gamma)
    pi: Policy = greedy_policy(V, env, gamma)
    env.render_v(V, pi)