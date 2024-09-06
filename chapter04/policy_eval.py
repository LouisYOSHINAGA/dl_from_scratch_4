if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]  # s
Action: TypeAlias = Literal[0, 1, 2, 3]  # a
Policy: TypeAlias = dict[State, tuple[Action, float]]  # pi(a|s)
Reward: TypeAlias = float  # r = r(s,a,s')
RewardFunction: TypeAlias = dict[tuple[State, Action, State], Reward]  # r(s,a,s')
DiscountRate: TypeAlias = float  # gamma
StateValueFunction: TypeAlias = dict[State, float]  # v(s)


def eval_onestep(pi: Policy, V: StateValueFunction, env: GridWorld, gamma: DiscountRate =0.9) -> StateValueFunction:
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
        else:
            new_V: float = 0
            action_probs: tuple[Action, float] = pi[state]
            for action, action_prob in action_probs.items():
                next_state: State = env.next_state(state, action)
                # V_{k+1}(s) = \sum_{a} pi(a|s) ( r(s,a,s') + gamma * V_{k}(s') )
                new_V += action_prob * (env.reward(state, action, next_state) + gamma * V[next_state])
            V[state] = new_V
    return V

def policy_eval(pi: Policy, V: StateValueFunction, env: GridWorld, gamma: DiscountRate,
                threshold: float=0.01) -> StateValueFunction:
    while True:
        old_V: StateValueFunction = V.copy()
        V: StateValueFunction = eval_onestep(pi, V, env, gamma)

        delta: float = 0  # max_{s}( V_{k+1}(s) - V_{k}(s) )
        for state in V.keys():
            t: float = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    """
    env = GridWorld()
    V = {}
    for state in env.states():
        V[state] = 0
    state: State = (1, 2)
    print(V[state])
    """
    """
    env = GridWorld()
    V: StateValueFunction = defaultdict(lambda: 0)
    state: State = (1, 2)
    print(V[state])
    """
    """
    pi: Policy = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    state: State = (0, 1)
    print(pi[state])
    """

    env = GridWorld()
    gamma: DiscountRate = 0.9
    pi: Policy = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V: StateValueFunction = defaultdict(lambda: 0)
    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)