if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from policy_eval import policy_eval
from common.gridworld import GridWorld

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]  # s
Action: TypeAlias = Literal[0, 1, 2, 3]  # a
Policy: TypeAlias = dict[State, tuple[Action, float]]  # pi(a|s)
Reward: TypeAlias = float  # r = r(s,a,s')
RewardFunction: TypeAlias = dict[tuple[State, Action, State], Reward]  # r(s,a,s')
DiscountRate: TypeAlias = float
StateValueFunction: TypeAlias = dict[State, float]  # v(s)
ActionValueFunction: TypeAlias = dict[tuple[State, Action], float]  # q(s,a)


def argmax(d: dict[Action, float]) -> Action:
    max_value: float = max(d.values())
    max_key: Action = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

def greedy_policy(V: StateValueFunction, env: GridWorld, gamma: DiscountRate) -> Policy:
    pi: Policy = {}
    for state in env.states():
        action_values: ActionValueFunction = {}
        for action in env.actions():
            next_state: State = env.next_state(state, action)  # s'
            action_values[action] = env.reward(state, action, next_state) + gamma * V[next_state]  # v(s') = r(s,a,s') + gamma * v(s')
        max_action: Action = argmax(action_values)
        pi[state] = {action: (1.0 if action == max_action else 0.0) for action in env.actions()}
    return pi

def policy_iter(env: GridWorld, gamma: DiscountRate, threshold: float =0.001, is_render: bool =False) -> Policy:
    pi: Policy = defaultdict(lambda: {action: 1/len(env.actions()) for action in env.actions()})
    V: StateValueFunction = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)  # V_{k} -> V_{k+1}
        new_pi: Policy = greedy_policy(V, env, gamma)  # pi -> pi'

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break
        else:
            pi = new_pi
    return pi


if __name__ == "__main__":
    env = GridWorld()
    gamma: DiscountRate= 0.9
    pi: Policy = policy_iter(env, gamma, is_render=True)