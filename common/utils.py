import numpy as np

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]
Action: TypeAlias = Literal[0, 1, 2, 3]
ActionValue: TypeAlias = float
ActionValueFunction: TypeAlias = dict[tuple[State, Action], ActionValue]


def greedy_probs(Q: ActionValueFunction, state: State,
                 epsilon: float =0, actions: list[Action] =[0, 1, 2, 3]) -> dict[Action, float]:
    max_action: Action = np.argmax([Q[state, action] for action in actions])
    action_probs: dict[Action, float] = {action: epsilon/len(actions) for action in actions}
    action_probs[max_action] += 1 - epsilon
    return action_probs