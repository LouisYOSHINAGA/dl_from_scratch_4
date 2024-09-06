from typing import TypeAlias, Literal

State: TypeAlias = Literal["L1", "L2"]
Action: TypeAlias = Literal["left", "right"]
Policy: TypeAlias = dict[tuple[State, Action], float]
Reward: TypeAlias = float
RewardFunction: TypeAlias = dict[tuple[State, Action, State], Reward]
DiscountRate: TypeAlias = float
StateValueFunction: TypeAlias = dict[State, float]


states: list[State] = ["L1", "L2"]  # states: s
actions: list[Action] = ["left", "right"]  # actions: a

pi: Policy = {
    # policy: p(a|s)
    ("L1", "left"): 0.5, ("L1", "right"): 0.5,
    ("L2", "left"): 0.5, ("L2", "right"): 0.5,
}
# state transition is deterministic: p(s'|s,a) -> s' = f(s,a)
r: RewardFunction = {
    # reward function: r(s,a,s')
    ("L1", "left", "L1"): -1, ("L1", "right", "L2"): +1,
    ("L2", "left", "L1"):  0, ("L2", "right", "L2"): -1,
}
gamma: DiscountRate = 0.9  # discount rate: gamma

V: StateValueFunction = {"L1": 0.0, "L2": 0.0}  # state value function: V_{k}(s)
new_V: StateValueFunction = V.copy()  # state value function: V_{k+1}(s)


if __name__ == "__main__":
    cnt: int = 0
    while True:
        deltas: dict[State, float] = {state: 0 for state in states}

        for state in states:
            new_V[state] = 0
            for action, next_state in zip(actions, states):
                # V_{k+1}(s) = \sum_{a} pi(a|s) ( r(s,a,s') + gamma * V_{k}(s') )
                new_V[state] += pi[(state, action)] * (r[(state, action, next_state)] + gamma * V[next_state])

            deltas[state] = abs(new_V[state] - V[state])
        V = new_V.copy()
        cnt += 1

        if max(deltas.values()) < 0.0001:
            break

    print(f"{V=}")
    print(f"{cnt}")