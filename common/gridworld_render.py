import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from typing import TypeAlias, Literal
State: TypeAlias = tuple[int, int]
Action: TypeAlias = Literal[0, 1, 2, 3]
Reward: TypeAlias = float
Policy: TypeAlias = dict[State, tuple[Action, float]]
StateValueFunction: TypeAlias = dict[State, float]
ActionValueFunction: TypeAlias = dict[tuple[State, Action], float]


class Renderer:
    def __init__(self, reward_map: np.ndarray, goal_state: State, wall_state: State) -> None:
        self.reward_map: np.ndarray = reward_map
        self.goal_state: State = goal_state
        self.wall_state: State = wall_state
        self.ys, self.xs = self.reward_map.shape
        self.actions: list[Action] = [0, 1, 2, 3]

        self.fig: matplotlib.figure.Figure|None = None
        self.ax: matplotlib.axes.Axes|None = None

    def set_figure(self, figsize: tuple[float, float]|None =None) -> None:
        fig = plt.figure(figsize=figsize)
        self.ax = fig.add_subplot(111)
        ax = self.ax
        ax.clear()
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.set_xticks(range(self.xs))
        ax.set_yticks(range(self.ys))
        ax.set_xlim(0, self.xs)
        ax.set_ylim(0, self.ys)
        ax.grid(True)

    def render_v(self, v: StateValueFunction|None =None, policy: Policy|None =None, print_value: bool =True) -> None:
        self.set_figure()
        xs: int = self.xs
        ys: int = self.ys
        ax: matplotlib.axes.Axes = self.ax

        if v is None:
            v_ndarray: np.ndarray|None = None
        else:
            v_dict: StateValueFunction = v
            v_ndarray = np.zeros(self.reward_map.shape)
            for state, value in v_dict.items():
                v_ndarray[state] = value
            vmin, vmax = v_ndarray.min(), v_ndarray.max()
            vmax = max(vmax, abs(vmin))
            vmin = -1 * vmax
            if vmax < 1:
                vmax = 1
            if vmin > -1:
                vmin = -1
            color_list: list[str] = ['red', 'white', 'green']
            cmap: matplotlib.colors.Colormap = matplotlib.colors.LinearSegmentedColormap.from_list('colormap_name', color_list)
            ax.pcolormesh(np.flipud(v_ndarray), cmap=cmap, vmin=vmin, vmax=vmax)

        for y in range(ys):
            for x in range(xs):
                state: State = (y, x)
                r: Reward = self.reward_map[y, x]
                if r != 0 and r is not None:
                    txt: str = 'R ' + str(r)
                    if state == self.goal_state:
                        txt = txt + ' (GOAL)'
                    ax.text(x+.1, ys-y-0.9, txt)

                if v_ndarray is not None and state != self.wall_state:
                    if print_value:
                        offsets: list[tuple[float, float]] = [(0.40, -0.15), (-0.15, -0.30)]
                        key: int = 0
                        if v_ndarray.shape[0] > 7:
                            key = 1
                        offset: tuple[float, float] = offsets[key]
                        ax.text(x+offset[0], ys-y+offset[1], f"{v[y, x]:12.2f}")

                if policy is not None and state != self.wall_state:
                    actions: dict[Action, float] = policy[state]
                    max_actions: Action = [k for k, v in actions.items() if v == max(actions.values())]

                    arrows: list[str] = ["↑", "↓", "←", "→"]
                    offsets: list[tuple[float, float]] = [(0.0, 0.1), (0.0, -0.1), (-0.1, 0.0), (0.1, 0.0)]
                    for action in max_actions:
                        if state == self.goal_state:
                            continue
                        ax.text(x+0.45+offsets[action][0], ys-y-0.5+offsets[action][1], arrows[action])

                if state == self.wall_state:
                    ax.add_patch(plt.Rectangle((x,ys-y-1), 1, 1, fc=(0.4, 0.4, 0.4, 1.0)))
        plt.show()

    def render_q(self, q: ActionValueFunction, show_greedy_policy: bool =True) -> None:
        self.set_figure()
        xs: int = self.xs
        ys: int = self.ys
        ax: matplotlib.axes.Axes = self.ax

        qmax, qmin = max(q.values()), min(q.values())
        qmax = max(qmax, abs(qmin))
        qmin = -1 * qmax
        if qmax < 1:
            qmax = 1
        if qmin > -1:
            qmin = -1

        color_list: list[str] = ['red', 'white', 'green']
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('colormap_name', color_list)

        for y in range(ys):
            for x in range(xs):
                for action in self.actions:
                    state: State = (y, x)
                    r: Reward = self.reward_map[y, x]
                    if r != 0 and r is not None:
                        txt: str = 'R ' + str(r)
                        if state == self.goal_state:
                            txt = txt + ' (GOAL)'
                        ax.text(x+0.05, ys-y-0.95, txt)
                    if state == self.goal_state:
                        continue

                    tx: int = x
                    ty: int = ys - y - 1
                    action_map: dict[Action, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {
                        0: ((0.5+tx, 0.5+ty), (1.0+tx, 1.0+ty), (    tx, 1.0+ty)),
                        1: ((    tx,     ty), (1.0+tx,     ty), (0.5+tx, 0.5+ty)),
                        2: ((    tx,     ty), (0.5+tx, 0.5+ty), (    tx, 1.0+ty)),
                        3: ((0.5+tx, 0.5+ty), (1.0+tx,     ty), (1.0+tx, 1.0+ty)),
                    }
                    offset_map: dict[Action, tuple[float, float]] = {
                        0: ( 0.1, 0.8),
                        1: ( 0.1, 0.1),
                        2: (-0.2, 0.4),
                        3: ( 0.4, 0.4),
                    }
                    if state == self.wall_state:
                        ax.add_patch(plt.Rectangle((tx, ty), 1, 1, fc=(0.4, 0.4, 0.4, 1.0)))
                    elif state in self.goal_state:
                        ax.add_patch(plt.Rectangle((tx, ty), 1, 1, fc=(0.0, 1.0, 0.0, 1.0)))
                    else:
                        tq: float = q[(state, action)]
                        color_scale: float = 0.5 + (tq / qmax) / 2  # normalize: 0.0-1.0
                        ax.add_patch(plt.Polygon(action_map[action], fc=cmap(color_scale)))
                        ax.text(tx+offset_map[action][0], ty+offset_map[action][1], f"{tq:12.2f}")
        plt.show()

        if show_greedy_policy:
            policy: Policy = {}
            for y in range(self.ys):
                for x in range(self.xs):
                    state: State = (y, x)
                    max_action = np.argmax([q[state, action] for action in self.actions])
                    probs: dict[Action, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
                    probs[max_action] = 1
                    policy[state] = probs
            self.render_v(None, policy)