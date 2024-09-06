import numpy as np
import common.gridworld_render as render_helper

from typing import TypeAlias, Literal, Generator

State: TypeAlias = tuple[int, int]  # s
Action: TypeAlias = Literal[0, 1, 2, 3]  # a
Policy: TypeAlias = dict[State, tuple[Action, float]]  # pi(a|s)
Reward: TypeAlias = int  # r(s,a,s')
StateValueFunction: TypeAlias = dict[State, float]  # v(s)
ActionValueFunction: TypeAlias = dict[tuple[State, Action], float]  # q(s,a)


class GridWorld:
    def __init__(self) -> None:
        self.action_space: list[Action] = [0, 1, 2, 3]
        self.action_meaning: dict[Action, str] = {
            0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT",
        }
        self.reward_map: np.ndarray = np.array([
            [0.0,  0.0, 0.0,  1.0],
            [0.0, None, 0.0, -1.0],
            [0.0,  0.0, 0.0,  0.0]
        ])
        self.wall_state: State = (1, 1)
        self.goal_state: State = (0, 3)
        self.start_state: State = (2, 0)
        self.agent_state: State = self.start_state

    @property
    def height(self) -> int:
        return self.reward_map.shape[0]

    @property
    def width(self) -> int:
        return self.reward_map.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.reward_map.shape

    # a
    def actions(self) -> list[Action]:
        return self.action_space

    # s
    def states(self) -> Generator[tuple[int, int], None, None]:
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    # p(s'|s,a) -> s' = f(s,a)
    def next_state(self, state: State, action: Action) -> State:
        action_move_map: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move: tuple[int, int] = action_move_map[action]
        next_state: State = tuple(s + m for s, m in zip(state, move))

        if next_state[0] < 0 or self.height <= next_state[0] \
           or next_state[1] < 0 or self.width <= next_state[1]:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state
        return next_state

    # r(s,a,s')
    def reward(self, state: State, action: Action, next_state: State) -> Reward:
        return self.reward_map[next_state]

    def reset(self) -> State:
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action: Action) -> tuple[State, Reward, bool]:
        state: State = self.agent_state
        next_state: State = self.next_state(state, action)
        reward: int = self.reward(state, action, next_state)
        done: bool = (next_state == self.goal_state)
        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, V: StateValueFunction|None =None, policy: Policy|None =None, print_value: bool =True) -> None:
        renderer = render_helper.Renderer(self.reward_map, self.goal_state, self.wall_state)
        renderer.render_v(V, policy, print_value)

    def render_q(self, q: ActionValueFunction|None =None, print_value: bool =True) -> None:
        renderer = render_helper.Renderer(self.reward_map, self.goal_state, self.wall_state)
        renderer.render_q(q, print_value)


if __name__ == "__main__":
    env = GridWorld()
    env.render_v()

    V: dict[State, float] = {}
    rng: np.random.Generator = np.random.default_rng()
    for state in env.states():
        V[state] = rng.standard_normal()
    env.render_v(V)